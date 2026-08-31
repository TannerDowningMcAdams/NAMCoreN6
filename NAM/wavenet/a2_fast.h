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

/// \brief How a layer's history ring is sized.
///
/// The ring holds max_lookback frames plus one block of new frames, and
/// write_pos advances one block per call. Sizing decides two things at once:
/// whether a block's write can straddle the end of the ring, and how much
/// history the model owns.
enum class RingPolicy
{
  /// max_lookback + one block, no rounding. Smallest possible history, and the
  /// only one where a write can straddle - which costs two runtime-length
  /// memcpys per channel row on the blocks where it happens. Measured at 4.0
  /// straddles per block per model across the 23 layers plus the head,
  /// concentrated in the shallow layers whose rings are shortest: dilation 1
  /// gives a 13-frame ring against an 8-frame block, so it straddles 5 blocks
  /// in 8.
  Exact = 0,
  /// Rounded up to a whole number of blocks, with write_pos kept block-aligned
  /// to match. write_pos is then always a multiple of the block size and
  /// wp + block <= ring_size always holds, so the straddle path is unreachable
  /// and every ring write is one compile-time-length copy per row. It also
  /// halves the tail-mirror refreshes, because `wrapped || wp < block`
  /// collapses to `wp == 0`. Costs about 1.3% more history, nearly all of it in
  /// the shallow layers - dilation 1 pads 13 to 16, dilation 239 pads 1203 to
  /// 1208.
  ///
  /// Not free of consequences elsewhere: it breaks the identity
  /// ring_size == max_lookback + block, which _layer_forward_fm2 relied on to
  /// reduce the deepest tap base to write_pos. Layer::read_base exists because
  /// of that; see the note there before adding a kernel that walks the ring.
  BlockAligned = 1
};

/// \brief Ring capacity in frames, under a policy.
///
/// constexpr and in the header because it has two consumers that cannot share
/// an implementation detail: the model sizes its rings from it at run time, and
/// the streaming interface below sizes host buffers from it at compile time.
/// One definition, so the two cannot drift - which is exactly how a ring got
/// mis-sized here once already.
constexpr int ring_frames(RingPolicy policy, int lookback, int block)
{
  const int exact = lookback + block;
  return (policy == RingPolicy::BlockAligned)
           ? ((exact + block - 1) / block) * block
           : exact;
}

/// \brief Where write_pos starts, under a policy.
///
/// Its own function because write_pos is reset in two places - when the rings
/// are sized, and again when a model is restored from its cached prewarm state.
/// A prewarmed model that came back mis-aligned would silently reintroduce the
/// straddles it was sized to avoid, which is a performance bug with no failure
/// mode to catch it.
constexpr int ring_start(RingPolicy policy, int lookback, int block)
{
  // Rounding up rather than down keeps it inside the ring, since
  // round_up(lookback) <= lookback + block - 1 < ring_frames().
  return (policy == RingPolicy::BlockAligned)
           ? ((lookback + block - 1) / block) * block
           : lookback;
}

/// \brief Select the ring sizing that the NEXT model construction will adopt.
///
/// Same contract as SetKernelForNextModel: sticky, read once at construction,
/// not thread-safe, call it during setup. Ring geometry is settled in
/// SetMaxBufferSize() and cannot change afterwards.
void SetRingPolicyForNextModel(RingPolicy p);

/// \brief Ring sizing most recently selected by SetRingPolicyForNextModel().
RingPolicy GetPendingRingPolicy();

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

/// \brief Kernel the most recently constructed model actually adopted, after
/// every downgrade has been applied.
///
/// The requested kernel and the adopted one differ silently, which is the one
/// failure this API cannot report by returning: a downgraded model builds,
/// runs, and produces correct output, just slowly and bit-identically to the
/// reference. A harness that records the request rather than this cannot tell
/// a working variant from a downgraded one.
Kernel GetLastModelKernel();

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

/// \brief Floats in one model's working-buffer block.
///
/// _layer_in, _head_sum and _z are Channels rows of max_buffer_size; _cond and
/// _head_out are one row each. All five are sized together and none carries
/// state between blocks, so they are one allocation rather than five.
constexpr int scratch_floats(int channels, int max_buffer_size)
{
  return 3 * channels * max_buffer_size // layer_in, head_sum, z
         + 2 * max_buffer_size;         // cond, head_out
}

/// \brief Somewhere other than the heap to put a model's working buffers.
///
/// The same argument as WeightArena, further along: these are ~350 bytes for a
/// 3-channel model at a block of 8, and every layer reads and writes them. By
/// access count they are the largest single consumer of the data cache in the
/// whole model -- larger than the ring histories, which are 200x their size but
/// are read a few hundred floats at a time. Highest traffic per byte in the
/// model, so the first thing worth pinning after the weights.
///
/// Sized from max_buffer_size, so unlike the weight block this is allocated in
/// SetMaxBufferSize() rather than in the constructor, and reallocated if the
/// block size changes.
struct ScratchArena
{
  /// Allocate `bytes`, aligned to at least 16. Return nullptr when full --
  /// the model then falls back to the heap rather than failing to build.
  void* (*alloc)(std::size_t bytes, void* ctx);
  /// Release a block previously returned by alloc(). Never called with null.
  void (*release)(void* block, void* ctx);
  void* ctx;
};

/// \brief Route working-buffer allocation for subsequently constructed models.
///
/// Sticky, like SetWeightArena(). Pass nullptr to go back to the heap. A model
/// captures the arena at construction and uses that same one to release, so
/// changing this later cannot strand an existing model's block.
void SetScratchArena(const ScratchArena* arena);

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
