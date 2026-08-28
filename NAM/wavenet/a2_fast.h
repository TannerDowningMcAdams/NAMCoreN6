#pragma once

// Specialized WaveNet fast path for the A2-Full (Channels=8) and
// A2-Lite (Channels=3) models. Shares the same architecture shape; only
// the channel count differs.
//
// When NAM_ENABLE_A2_FAST is defined at build time, wavenet::create_config
// consults is_a2_shape() on every incoming WaveNet config and, on match,
// instantiates an A2FastModel<Channels> instead of the generic WaveNet.
//
// The baseline here is correct-but-unoptimized (plain column-major loops).
// Follow-up optimizations (unrolled GEMV, tap-major nest, factored
// per-kernel-size helpers) plug into the same class.

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
/// The JSON overload above can only serve loaders that have a JSON document.
/// A binary loader (.namb) builds a WaveNetConfig directly and hands it to
/// create_dsp(), which does no shape checking -- so without this overload an
/// A2 model loaded from .namb silently runs the generic WaveNet.
///
/// This inspects the typed parameters the model will actually be built from,
/// rather than re-deriving the JSON predicates in binary terms. That keeps the
/// two entry points from drifting: a field the binary format gains later cannot
/// bypass the check, because the check reads the constructed config.
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

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
