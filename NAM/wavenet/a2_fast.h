#pragma once

// Specialized WaveNet fast path for the A2-Full (Channels=8) and
// A2-Lite (Channels=3) models. Shares the same architecture shape; only
// the channel count differs.
//
// When NAM_ENABLE_A2_FAST is defined at build time, wavenet::create_config
// consults is_a2_shape() on every incoming WaveNet config and, on match,
// instantiates an A2FastModel<Channels> instead of the generic WaveNet.

#if defined(NAM_ENABLE_A2_FAST)

  #include <array>
  #include <memory>

  #include "../model_config.h"
  #include "json.hpp"
  #include "model.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

/// \brief Number of layers in an A2 layer array.
constexpr int kNumLayers = 23;
/// \brief Kernel size of the layer-array head rechannel convolution.
constexpr int kHeadKernelSize = 16;
/// \brief LeakyReLU negative-slope used by every layer.
constexpr float kLeakySlope = 0.01f;

/// \brief Per-layer kernel sizes (fixed pattern shared by A2-Full + A2-Lite).
inline constexpr std::array<int, kNumLayers> kKernelSizes = {
  6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 15, 15, 6, 6, 6, 6, 6, 6, 6};

/// \brief Per-layer dilations (fixed pattern shared by A2-Full + A2-Lite).
inline constexpr std::array<int, kNumLayers> kDilations = {
  1, 3, 7, 17, 41, 101, 239, 1, 3, 7, 17, 41, 101, 239, 1, 13, 1, 3, 7, 17, 41, 101, 239};

/// \brief Strict detector: returns true iff config matches the A2 shape.
/// \param config   The "config" sub-object from a .nam WaveNet entry.
/// \param channels Out-param set to 3 (A2-Lite) or 8 (A2-Full) on match.
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape(const nlohmann::json& config, int* channels);

/// \brief Strict detector operating on an already-parsed WaveNetConfig.
///
/// A binary loader (.namb) builds a WaveNetConfig directly and hands it to
/// create_dsp(), which does no shape checking, so without this overload an A2
/// model from .namb silently runs the generic WaveNet. Inspecting the
/// constructed config rather than re-deriving the JSON predicates keeps the two
/// entry points from drifting as the binary format gains fields.
///
/// \param config   Parsed WaveNet configuration.
/// \param channels Out-param set to 3 (A2-Lite) or 8 (A2-Full) on match.
/// \return true if every architectural knob matches the A2 signature exactly.
bool is_a2_shape(const WaveNetConfig& config, int* channels);

/// \brief Build a ModelConfig that instantiates the A2 fast path.
/// \pre is_a2_shape(config, ...) returned true.
std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate);

/// \brief Build a ModelConfig that instantiates the A2 fast path.
/// \param channels 3 (A2-Lite) or 8 (A2-Full), as reported by is_a2_shape().
/// \pre is_a2_shape() returned true and produced this channel count.
std::unique_ptr<ModelConfig> create_a2_fast_config(int channels);

/// \brief Per-layer kernel implementation.
///
/// The history layout follows from this, so it is fixed when the model is
/// constructed and cannot be changed afterwards.
enum class Kernel
{
  /// Scalar, column-major history. The reference implementation and the
  /// golden output that the others are checked against.
  Reference = 0,
  /// Helium, frame-major (transposed) history. Requires Channels == 3, a
  /// compile-time block size that is a multiple of 4, and MVE float support;
  /// falls back to Reference when any of those is missing.
  MveFrameMajor = 1,
  /// As MveFrameMajor, same layout and same math, but both 4-frame groups are
  /// carried through the tap loop so each weight load feeds two FMAs and the
  /// tap addressing is computed once instead of twice. Requires a block of
  /// exactly 8 frames.
  MveFrameMajorWide = 2
};

/// \brief Select the kernel that the NEXT model construction will adopt.
///
/// Exists so a host can build one model per kernel and compare them on target
/// for both cycles and output. Sticky: it applies to every subsequent
/// construction until changed. Not thread-safe -- call it during setup.
///
/// Requests that the build cannot satisfy are silently downgraded to
/// Reference, so the caller should read back GetKernel() on the model it got
/// rather than assume. There is no way to ask an existing model to change.
void SetKernelForNextModel(Kernel k);

/// \brief Kernel most recently selected by SetKernelForNextModel().
/// Note this is the *requested* kernel, not necessarily what a given model
/// adopted -- see the downgrade note above.
Kernel GetPendingKernel();

/// \brief Floats in one model's weight block, for a given channel count.
///
/// Every weight lives in one contiguous block: the rechannel vector, then per
/// layer the dilated conv kernel, its bias, the input mixin and the 1x1 with
/// its bias, then the head kernel. head_bias and head_scale stay in the object.
///
/// Constant at compile time because is_a2_shape() admits exactly one geometry -
/// captures differ in weight values, never in shape - so a host can size a pool
/// with a static_assert rather than a guess.
constexpr int weight_block_floats(int channels)
{
  int n = channels; // rechannel
  for (int i = 0; i < kNumLayers; i++)
  {
    n += kKernelSizes[i] * channels * channels // conv kernel
         + channels                            // conv bias
         + channels                            // input mixin
         + channels * channels                 // layer 1x1
         + channels;                           // layer 1x1 bias
  }
  n += kHeadKernelSize * channels; // head kernel
  return n;
}

/// \brief Somewhere other than the heap to put model weights.
///
/// The weights are ~8% of a model's bytes but roughly 30% of its per-block
/// memory traffic, which makes them worth pinning in fast memory when the ring
/// histories cannot be. One alloc and one release per model, so a fixed-slot
/// pool is a sufficient implementation.
struct WeightArena
{
  /// Allocate `bytes`, aligned to at least 16. Return nullptr when full --
  /// the model then falls back to the heap rather than failing to build.
  void* (*alloc)(std::size_t bytes, void* ctx);
  /// Release a block previously returned by alloc(). Never called with null.
  void (*release)(void* block, void* ctx);
  void* ctx;
};

/// \brief Route weight allocation for subsequently constructed models.
///
/// Sticky, like SetKernelForNextModel(). Pass nullptr to go back to the heap.
/// A model captures the arena at construction and uses that same one to
/// release, so changing this later cannot strand an existing model's block.
void SetWeightArena(const WeightArena* arena);

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
