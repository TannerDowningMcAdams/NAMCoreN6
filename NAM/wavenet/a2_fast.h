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

// -----------------------------------------------------------------------------
// History streaming.
//
// A layer's per-block reads are K windows of `frames_per_block` consecutive
// frames at stride `dilation`, once per channel row -- a few hundred floats out
// of a ring that runs to 9.5 KB. Caching that is hopeless; gathering it is not.
// A host that can gather (a DMA engine, or just a copy loop) can therefore keep
// the ring in slow memory and hand the model a compact, fast-memory copy of
// only the windows it will read.
//
// Which layers qualify is fixed by the geometry: a layer whose dilation is at
// least the block size has every tap but the newest at least one block old, and
// the newest is peeled out of the kernel and read from the layer input. So the
// gather for a block depends only on writes completed before that block began,
// and can be issued at the top of the block with a whole block to land.
//
// Layers below that threshold keep their history in the model's own memory and
// run the existing kernel. So does every layer when no streamer is installed,
// or when the streamer refuses one -- streaming is an optimisation the host may
// decline per layer, never a requirement.
// -----------------------------------------------------------------------------

/// \brief Ring capacity for a streamed layer, in frames.
///
/// Rounded up to a whole number of blocks so write_pos only ever lands on a
/// block boundary, which keeps every ring write contiguous.
constexpr int streamed_ring_frames(int layer, int frames_per_block)
{
  const int lookback = (kKernelSizes[layer] - 1) * kDilations[layer];
  const int exact = lookback + frames_per_block;
  return ((exact + frames_per_block - 1) / frames_per_block) * frames_per_block;
}

/// \brief Floats per channel row of a streamed layer's history.
///
/// Twice the ring: the second copy mirrors the first, so the span a block reads
/// is one contiguous run at every write position and the gather is a single
/// fixed-stride descriptor rather than a wrap-dependent pair.
constexpr int streamed_row_floats(int layer, int frames_per_block)
{
  return 2 * streamed_ring_frames(layer, frames_per_block);
}

/// \brief True if this layer's taps are all at least one block old.
constexpr bool layer_is_streamable(int layer, int frames_per_block)
{
  return kDilations[layer] >= frames_per_block;
}

/// \brief Floats the host must supply for one model's streamed histories.
constexpr int streamed_history_floats(int channels, int frames_per_block)
{
  int n = 0;
  for (int i = 0; i < kNumLayers; i++)
    if (layer_is_streamable(i, frames_per_block))
      n += channels * streamed_row_floats(i, frames_per_block);
  return n;
}

/// \brief Floats one streamed layer's gathered windows occupy.
///
/// K-1 windows, not K: the newest tap comes from the layer input directly.
/// Laid out channel-major then tap-major, so the gather writes it linearly.
constexpr int layer_tap_floats(int layer, int channels, int frames_per_block)
{
  return channels * (kKernelSizes[layer] - 1) * frames_per_block;
}

/// \brief Largest layer_tap_floats() over the streamed layers. Slot size for a
/// host that recycles a few staging buffers rather than holding all of them.
constexpr int max_layer_tap_floats(int channels, int frames_per_block)
{
  int m = 0;
  for (int i = 0; i < kNumLayers; i++)
    if (layer_is_streamable(i, frames_per_block))
    {
      const int n = layer_tap_floats(i, channels, frames_per_block);
      if (n > m) m = n;
    }
  return m;
}

/// \brief layer_tap_floats() summed over every streamed layer: the staging cost
/// of holding a whole block's gather at once.
constexpr int block_tap_floats(int channels, int frames_per_block)
{
  int n = 0;
  for (int i = 0; i < kNumLayers; i++)
    if (layer_is_streamable(i, frames_per_block))
      n += layer_tap_floats(i, channels, frames_per_block);
  return n;
}

/// \brief Streamed layers in a model, for sizing per-layer host tables.
constexpr int streamed_layer_count(int frames_per_block)
{
  int n = 0;
  for (int i = 0; i < kNumLayers; i++)
    if (layer_is_streamable(i, frames_per_block)) n++;
  return n;
}

/// \brief What the host needs to know to gather one layer's windows.
///
/// Pure geometry: the model states where its ring is and how it is shaped, and
/// the host decides how to move the bytes. Fixed for the life of the binding.
struct StreamGeometry
{
  int layer;            ///< Index into kKernelSizes / kDilations.
  int channels;         ///< Rows in the history, and in the gathered block.
  int taps;             ///< Windows to gather: kernel_size - 1.
  int dilation;         ///< Frames between consecutive window bases.
  int frames_per_block; ///< Floats per window.
  int ring_frames;      ///< streamed_ring_frames(): write_pos wraps at this.
  int row_floats;       ///< streamed_row_floats(): stride between channel rows.
  float* history;       ///< Base of the model's ring, from alloc_history().
};

/// \brief Opaque host handles. The model stores them and passes them back.
using ModelToken = void*;
using StreamSlot = void*;

/// \brief Somewhere other than the model's own memory to put ring histories,
/// and something to gather from them.
///
/// Follows WeightArena: installed globally, captured at construction, and free
/// to refuse. A model that gets nothing back behaves exactly as if no streamer
/// were installed.
struct HistoryStreamer
{
  /// Claim whatever the host tracks per model - a DMA channel, a staging pool.
  /// Return nullptr to decline the whole model.
  ModelToken (*open_model)(void* ctx);
  /// Release a token from open_model(). Never called with nullptr.
  void (*close_model)(ModelToken model, void* ctx);

  /// Ring storage for one streamed layer, aligned to at least 16 and zeroed.
  /// Return nullptr to decline this layer.
  float* (*alloc_history)(std::size_t floats, ModelToken model, void* ctx);
  /// Release a block from alloc_history(). Never called with nullptr.
  void (*free_history)(float* block, ModelToken model, void* ctx);

  /// Accept responsibility for gathering one layer. Return nullptr to decline
  /// it; the model then reads in place the ring alloc_history() gave it, so
  /// declining costs correctness nothing.
  StreamSlot (*bind)(const StreamGeometry& geometry, ModelToken model, void* ctx);
  /// Release a slot from bind(). Never called with nullptr.
  void (*unbind)(StreamSlot slot, ModelToken model, void* ctx);

  /// Start the gather for one block. `write_pos` holds one entry per streamed
  /// layer, in bind() order, each the layer's ring write position BEFORE this
  /// block's write -- which is what makes every gathered window at least one
  /// block old. Called once, before any layer runs.
  void (*prefetch)(ModelToken model, const int* write_pos, void* ctx);

  /// The gathered windows for one layer, laid out channel-major then tap-major,
  /// oldest tap first. Blocks until they are there. Never returns nullptr after
  /// a successful bind().
  const float* (*taps)(StreamSlot slot, void* ctx);

  void* ctx;
};

/// \brief Route history allocation and gathering for subsequently constructed
/// models. Sticky, like SetWeightArena(). Pass nullptr to go back to holding
/// every history in the model's own memory.
void SetHistoryStreamer(const HistoryStreamer* streamer);

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
