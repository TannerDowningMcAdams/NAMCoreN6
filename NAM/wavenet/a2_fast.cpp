#if defined(NAM_ENABLE_A2_FAST)

  // Ring-buffer strategy:
  //   0 = linear memmove-rewind (variable worst-case latency, sporadic spikes)
  //   1 = pow2 + tail mirror (constant per-block work, branchless reads)
  // Controlled externally with -DNAM_A2_RING_MODE=0 for head-to-head comparison.
  #ifndef NAM_A2_RING_MODE
    #define NAM_A2_RING_MODE 1
  #endif

  // Compile-time model block size, in frames. 0 = dynamic (upstream behaviour:
  // the block size is whatever Reset()/SetMaxBufferSize() was given).
  //
  // Setting it turns num_frames and GetMaxBufferSize() into compile-time
  // constants through the whole hot path, which buys two things on Cortex-M
  // that matter more the smaller the block is:
  //
  //   - the std::memcpy sizes in the ring writes become constants, so GCC
  //     expands them inline instead of calling newlib's memcpy. At a 8-frame
  //     block that is 48 out-of-line calls per block, each moving <= 96 bytes,
  //     and newlib's generic memcpy is poor on this core.
  //
  //   - the per-frame loops become constant-trip, so GCC can fully unroll them
  //     and keep the z accumulator in registers across taps instead of
  //     reloading and restoring it from _z on every tap.
  //
  // A model whose block size does not match is refused in process() rather
  // than run: every bound below would be wrong.
  #ifndef NAM_A2_FIXED_BLOCK
    #define NAM_A2_FIXED_BLOCK 0
  #endif

  // Optional external profiling probes, off by default.
  //
  // This library must not depend on firmware headers, so the probe is a pair of
  // extern "C" symbols the host image supplies. Enable with -DNAM_A2_PROFILE=1
  // and define both; leave it off and the calls vanish entirely.
  //
  // Probe ids are deliberately coarse. Per-layer probing costs 23x as much and
  // says nothing the aggregate does not: the accumulators on the host side sum
  // across calls, so bracketing every layer already yields the total share.
  #ifndef NAM_A2_PROFILE
    #define NAM_A2_PROFILE 0
  #endif

  // Which per-layer kernel a model uses. Values match a2_fast.h's Kernel enum.
  //
  //   0  Reference. Scalar, column-major history (frame-contiguous columns of
  //      Channels floats). The original path; the golden output.
  //   1  MveFrameMajor. Helium, frame-major history (channel-contiguous rows of
  //      frames), so a vector load gets 4 consecutive frames of one channel and
  //      the conv becomes vector-by-scalar FMAs with the accumulator resident
  //      in Q registers across all K taps. Channels == 3 only.
  //
  // This is the compile-time default. With NAM_A2_BENCH the host can override
  // it per model via SetKernelForNextModel(), which is how the two are compared
  // head to head on target.
  #ifndef NAM_A2_KERNEL_DEFAULT
    #define NAM_A2_KERNEL_DEFAULT 0
  #endif

  // __ARM_FEATURE_MVE bit 1 is the floating-point half of Helium. Bit 0 alone
  // (integer MVE) is not enough for any of this.
  #if defined(__ARM_FEATURE_MVE) && ((__ARM_FEATURE_MVE) & 2)
    #define NAM_A2_HAVE_MVE 1
    #include <arm_mve.h>
  #else
    #define NAM_A2_HAVE_MVE 0
  #endif

  #if NAM_A2_PROFILE
/// Probe ids passed to the host's nam_a2_profile_* hooks.
enum
{
  /// The K-specialized per-layer kernel only -- conv, mixin, activation,
  /// head-sum accumulate and the 1x1 residual. Excludes the ring write, so
  /// (model total - this) isolates the ring/copy overhead.
  kNamA2ProbeLayerMath = 0
};
extern "C" void nam_a2_profile_enter(int id);
extern "C" void nam_a2_profile_exit(int id);
    #define NAM_A2_PROF_ENTER(id) ::nam_a2_profile_enter(id)
    #define NAM_A2_PROF_EXIT(id) ::nam_a2_profile_exit(id)
  #else
    #define NAM_A2_PROF_ENTER(id) ((void)0)
    #define NAM_A2_PROF_EXIT(id) ((void)0)
  #endif

  #include "a2_fast.h"

  #include <algorithm>
  #include <array>
  #include <cmath>
  #include <cstddef>
  #include <cstring>
  #include <iterator>
  #include <memory>
  #if !defined(NAM_NO_EXCEPTIONS)
    #include <sstream>
  #endif
  #include <stdexcept>
  #include <string>
  #include <utility>
  #include <vector>

  #include <Eigen/Dense>

  #include "../dsp.h"
  #include "../status.h"

namespace nam
{
namespace wavenet
{
namespace a2_fast
{

namespace
{

/// Kernel that the next A2FastModel construction will adopt. Set through the
/// public SetKernelForNextModel(); read once, in the constructor. Not
/// thread-safe and not meant to be -- models are built during setup.
int g_pending_kernel = NAM_A2_KERNEL_DEFAULT;

// =============================================================================
// A2FastModel<Channels>
//
// Skeleton implementation: correct but not yet optimized.
//
// Architectural invariants (checked once by is_a2_shape before we get here):
//   - single layer array with 23 layers
//   - Bottleneck == Channels
//   - condition_size == input_size == out_channels == 1
//   - LeakyReLU(0.01) on every layer, no gating, no FiLM, no head1x1
//   - layer1x1 active (groups=1), head rechannel conv k=16 bias=true
//   - no post-stack head
//
// Weight storage: column-major per kernel tap. For a (out_ch × in_ch) matrix
// at tap k, element (row=i, col=j) lives at w[k][j * out_ch + i]. All 1×1
// and K×1 convolutions follow the same convention (with K = 1 for 1×1).
// =============================================================================
template <int Channels>
class A2FastModel : public DSP
{
public:
  static constexpr int kChannels = Channels;
  static constexpr int kBottleneck = Channels;
  static constexpr int kHeadIn = Channels;
  #if NAM_A2_FIXED_BLOCK > 0
  /// Block size this translation unit is specialized for. See NAM_A2_FIXED_BLOCK.
  static constexpr int kFixedBlock = NAM_A2_FIXED_BLOCK;
  #endif

  A2FastModel(std::vector<float> weights, double expected_sample_rate);
  ~A2FastModel() override = default;

  void process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames) override;
  void prewarm() override;
  int GetPrewarmSamples() override { return _prewarm_samples; }

protected:
  void SetMaxBufferSize(int maxBufferSize) override;

private:
  struct Layer
  {
    int kernel_size = 0;
    int dilation = 0;
    int max_lookback = 0; // (kernel_size - 1) * dilation

    // Dilated conv (Channels -> Bottleneck), column-major per tap.
    // Flat size = kernel_size * Channels * Bottleneck.
    std::vector<float> conv_w;
    std::array<float, Channels> conv_b{};

    // Input mixin (cond_size=1 -> Bottleneck), no bias.
    std::array<float, Channels> mixin_w{};

    // layer1x1 (Bottleneck -> Channels), with bias. Column-major (Channels × Bottleneck).
    std::array<float, Channels * Channels> l1x1_w{};
    std::array<float, Channels> l1x1_b{};

    // Conv1D input history ring buffer.
    //
    // Reference kernel: column-major -- (pow2_size + mbs) columns of Channels
    // contiguous floats, so one column is one frame across all channels.
    //
    // MveFrameMajor kernel: transposed -- Channels rows of row_stride floats,
    // so row c holds consecutive frames of channel c and four of them come in
    // on one vector load. Total allocation is identical either way; only the
    // interpretation differs, which is why SetMaxBufferSize() needs no branch.
    std::vector<float> history;
    int row_stride = 0; // frame-major only: pow2_size + max_buffer_size
    std::array<float, Channels> cached_prewarm_state{};
  #if NAM_A2_RING_MODE == 1
    // pow2 ring + tail mirror. Storage = (pow2_size + max_buffer_size) cols.
    // write_pos is kept in [0, pow2_size), reads use (pos & pow2_mask) and are
    // always contiguous because cols [pow2_size, pow2_size + max_buffer_size)
    // mirror cols [0, max_buffer_size).
    int ring_size = 0;
    int write_pos = 0;
  #else
    // Linear ring with sporadic memmove-rewind. history_cols = 2*max_lookback +
    // max_buffer_size; write_pos grows monotonically until rewind fires.
    int history_cols = 0;
    int write_pos = 0;
  #endif
  };

  std::array<Layer, kNumLayers> _layers;

  // Rechannel (input_size=1 -> Channels), no bias.
  std::array<float, Channels> _rechannel_w{};

  // Head rechannel (Bottleneck -> 1), kernel=16, bias. Column-major per tap.
  // At each tap, matrix is (1 × Channels) col-major -> Channels floats.
  std::array<std::array<float, Channels>, kHeadKernelSize> _head_w{};
  float _head_b = 0.0f;

  // Head scale is stored as the trailing float in the weights stream (the generic
  // WaveNet reads it the same way, overriding the JSON head_scale field).
  float _head_scale = 1.0f;

  // Head ring buffer (Channels rows, col-major). Same ring layout as per-layer.
  std::vector<float> _head_history;
  std::array<float, Channels> _cached_head_prewarm_state{};
  #if NAM_A2_RING_MODE == 1
  int _head_ring_size = 0;
  /// Frame-major only: floats per channel row of _head_history.
  int _head_row_stride = 0;
  int _head_write_pos = 0;
  #else
  int _head_history_cols = 0;
  int _head_write_pos = 0;
  #endif

  // Working buffers (all Channels rows, max_buffer_size cols, col-major).
  std::vector<float> _layer_in; // current layer input / next layer input (in-place residual)
  std::vector<float> _head_sum; // accumulates activations across all layers
  std::vector<float> _z; // per-layer conv output accumulator (tap-major)
  std::vector<float> _cond; // float32 copy of the double NAM_SAMPLE input, reused each block
  std::vector<float> _head_out; // float32 head output before writing to NAM_SAMPLE

  int _prewarm_samples = 0;
  bool _has_cached_prewarm_state = false;

  void _load_weights(std::vector<float>& weights);
  bool HasCachedPrewarmState() const { return _has_cached_prewarm_state; }
  void PrewarmFromCache();
  void CacheStateAsPrewarmed();
  void _ring_write(Layer& L, int num_frames);
  void _head_ring_write(int num_frames);
  void _layer_forward(int layer_idx, const float* cond, int num_frames);
  void _head_forward(float* output, int num_frames);

  /// Which per-layer kernel this instance runs. See NAM_A2_KERNEL_DEFAULT.
  /// Fixed at construction: the history layout depends on it, so it cannot
  /// change once buffers are sized and warmed.
  int _kernel = NAM_A2_KERNEL_DEFAULT;
  bool _frame_major() const { return _kernel == 1 || _kernel == 2; }

  #if NAM_A2_HAVE_MVE
  void _ring_write_fm(Layer& L, int num_frames);
  template <int KernelSize>
  void _layer_forward_fm(Layer& L, const float* cond, int num_frames);
  template <int KernelSize>
  void _layer_forward_fm2(Layer& L, const float* cond, int num_frames);
  void _head_ring_write_fm(int num_frames);
  void _head_forward_fm(float* output, int num_frames);
  #endif

  // Compile-time-specialized per-layer kernel. KernelSize is lifted to a
  // template parameter so clang can fully unroll the tap loop and schedule
  // FMAs across taps. For the A2 shape we only need K=6 and K=15.
  template <int KernelSize>
  void _layer_forward_k(Layer& L, const float* cond, int num_frames);
};

// -----------------------------------------------------------------------------
// Construction
// -----------------------------------------------------------------------------
template <int Channels>
A2FastModel<Channels>::A2FastModel(std::vector<float> weights, double expected_sample_rate)
: DSP(/*in_channels=*/1, /*out_channels=*/1, expected_sample_rate)
{
  // Latch the kernel choice for this instance now: the history layout follows
  // from it, so it must be settled before SetMaxBufferSize() sizes anything.
  _kernel = g_pending_kernel;
  #if !NAM_A2_HAVE_MVE
  _kernel = 0; // no Helium float on this target
  #endif
  #if NAM_A2_FIXED_BLOCK <= 0
  // The frame-major kernel needs a compile-time block size: it vectorizes four
  // frames at a time and sizes a transpose scratch from it.
  _kernel = 0;
  #endif
  if constexpr (Channels != 3)
  {
    // The frame-major kernel is written against a 3-channel layer. A2-Full
    // keeps the Eigen path, which already blocks over whole frames.
    _kernel = 0;
  }

  for (int i = 0; i < kNumLayers; i++)
  {
    _layers[i].kernel_size = kKernelSizes[i];
    _layers[i].dilation = kDilations[i];
    _layers[i].max_lookback = (kKernelSizes[i] - 1) * kDilations[i];
    _layers[i].conv_w.assign(static_cast<size_t>(kKernelSizes[i]) * Channels * Channels, 0.0f);
  }

  _load_weights(weights);

  // Receptive field = 1 (the sample being produced) + sum of per-layer lookbacks +
  // (head kernel - 1). The leading 1 matches the generic WaveNet's prewarm count
  // (model.cpp: mPrewarmSamples starts at 1 when there's no condition DSP), so the
  // fast path warms up by exactly the same number of samples as the model it replaces.
  int prewarm = 1;
  for (int i = 0; i < kNumLayers; i++)
    prewarm += _layers[i].max_lookback;
  prewarm += kHeadKernelSize - 1;
  _prewarm_samples = prewarm;
}

// -----------------------------------------------------------------------------
// Weight loader
//
// Reproduces the generic path's weight-reading order exactly:
//   - LayerArray::set_weights_:
//       _rechannel (Conv1x1 1 -> Channels, no bias)
//       for each layer:
//           _conv (Conv1D Channels -> Bottleneck, K × C × B + B bias)
//           _input_mixin (Conv1x1 1 -> Bottleneck, no bias)
//           _layer1x1 (Conv1x1 Bottleneck -> Channels, with bias)
//       _head_rechannel (Conv1D Bottleneck -> 1, K=16, bias)
//
// Generic Conv1D loader order: for i in out_ch: for j in in_ch: for k in taps.
// Generic Conv1x1 loader order: for i in out_ch: for j in in_ch.
// We permute into column-major per-tap storage while reading.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_load_weights(std::vector<float>& weights)
{
  auto it = weights.begin();
  const auto end = weights.end();

  auto take = [&]() -> float {
    if (it == end)
    {
      // Latched rather than thrown: this runs inside the constructor, which has
      // no way to report failure. A2FastConfig::create() checks the latch once
      // construction returns and discards the model.
      NAM_FAIL_RET(nam::Status::ErrorWeightCount, "A2FastModel: weight stream exhausted", 0.0f);
    }
    return *it++;
  };

  // Rechannel: 1 -> Channels, no bias. Read order: for i in Channels: for j in 1.
  for (int i = 0; i < Channels; i++)
    _rechannel_w[i] = take();

  for (int li = 0; li < kNumLayers; li++)
  {
    Layer& L = _layers[li];
    const int K = L.kernel_size;

    // Conv1D: Channels -> Bottleneck, kernel=K, bias.
    // Read order: for i in Bottleneck: for j in Channels: for k in K.
    // Store at conv_w[k * C * B + j * B + i] (col-major (B × C) per tap).
    for (int i = 0; i < Channels; i++) // row (out)
    {
      for (int j = 0; j < Channels; j++) // col (in)
      {
        for (int k = 0; k < K; k++)
        {
          L.conv_w[k * Channels * Channels + j * Channels + i] = take();
        }
      }
    }
    for (int i = 0; i < Channels; i++)
      L.conv_b[i] = take();

    // Input mixin: 1 -> Bottleneck, no bias. Read order: for i in Bottleneck: for j in 1.
    for (int i = 0; i < Channels; i++)
      L.mixin_w[i] = take();

    // layer1x1: Bottleneck -> Channels, with bias. Read order: for i in Channels: for j in Bottleneck.
    // Store at l1x1_w[j * Channels + i] (col-major Channels × Bottleneck).
    for (int i = 0; i < Channels; i++) // row (out = Channels)
    {
      for (int j = 0; j < Channels; j++) // col (in = Bottleneck)
      {
        L.l1x1_w[j * Channels + i] = take();
      }
    }
    for (int i = 0; i < Channels; i++)
      L.l1x1_b[i] = take();
  }

  // Head rechannel: Bottleneck -> 1, kernel=16, bias.
  // Read order: for i in 1: for j in Bottleneck: for k in 16.
  // Store at _head_w[k][j] (row=0 since out=1, column-major => just Channels floats per tap).
  for (int j = 0; j < Channels; j++)
  {
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      _head_w[k][j] = take();
    }
  }
  _head_b = take();

  // Matches WaveNet::set_weights_: the last value in the stream is head_scale.
  _head_scale = take();

  if (it != end)
  {
  #if defined(NAM_NO_EXCEPTIONS)
    // The message is discarded in this build, so do not build one: <sstream>
    // would otherwise be linked in purely to format text nobody reads.
    NAM_FAIL(nam::Status::ErrorWeightCount, "");
  #else
    std::stringstream ss;
    ss << "A2FastModel: weight stream has " << std::distance(it, end) << " trailing bytes";
    NAM_FAIL(nam::Status::ErrorWeightCount, ss.str());
  #endif
  }
}

// -----------------------------------------------------------------------------
// Buffer sizing
// -----------------------------------------------------------------------------
namespace
{
// Ring index wrapping for an exactly-sized (non power-of-two) ring.
//
// A conditional add/subtract rather than %, and a single one is always enough
// because every caller is at most one ring away:
//   - forward:  write_pos < size and the step is at most one block, itself <=
//     size, so the sum is < 2*size.
//   - backward: the deepest tap is write_pos - nf - max_lookback, and
//     size == max_lookback + nf exactly, so the value is >= -size.
// Costs one compare and one conditional op per tap against the mask it
// replaces. That is ~156 extra instructions per block, against the ~19.5k
// cycles of AXISRAM stall that fitting in DTCM removes.
inline int wrap_fwd(int v, int size)
{
  return (v >= size) ? (v - size) : v;
}

inline int wrap_back(int v, int size)
{
  return (v < 0) ? (v + size) : v;
}

// Fixed-length non-overlapping float copy.
//
// Not std::memcpy: even with a compile-time constant length, GCC declines to
// expand a 96-byte copy inline on this target and emits a call to newlib's
// memcpy, which is poor on Cortex-M -- and at an 8-frame block the ring writes
// would make ~48 such calls per block, all of them 96 bytes. Written as a
// constant-trip loop over floats it is fully unrolled into load/store pairs
// with no call, no alignment dispatch, and no size test.
//
// Callers must guarantee the ranges do not overlap; both ring-write uses do.
template <int N>
inline void copy_floats(float* __restrict dst, const float* __restrict src)
{
  for (int i = 0; i < N; i++)
    dst[i] = src[i];
}

/// One block's worth of a single channel row: nf floats. Unrolled when the
/// block size is compile-time. Ranges must not overlap.
inline void copy_row(float* dst, const float* src, int nf)
{
  #if NAM_A2_FIXED_BLOCK > 0
  (void)nf;
  copy_floats<NAM_A2_FIXED_BLOCK>(dst, src);
  #else
  std::memcpy(dst, src, static_cast<size_t>(nf) * sizeof(float));
  #endif
}

/// One block's worth of columns: nf * Channels floats. Unrolled when the block
/// size is compile-time, a plain memcpy otherwise. Ranges must not overlap.
template <int Channels>
inline void copy_block(float* dst, const float* src, int nf)
{
  #if NAM_A2_FIXED_BLOCK > 0
  (void)nf; // == NAM_A2_FIXED_BLOCK, enforced in process()/SetMaxBufferSize()
  copy_floats<NAM_A2_FIXED_BLOCK * Channels>(dst, src);
  #else
  std::memcpy(dst, src, static_cast<size_t>(nf) * Channels * sizeof(float));
  #endif
}
} // namespace

template <int Channels>
void A2FastModel<Channels>::SetMaxBufferSize(int maxBufferSize)
{
  #if NAM_A2_FIXED_BLOCK > 0
  // Caught here, at Reset() time, where A2FastConfig::create()'s caller can
  // still see the latch and discard the model -- rather than at the first
  // process() call, in the audio interrupt, with the buffers already sized
  // wrong. Everything below allocates to kFixedBlock regardless.
  if (maxBufferSize != kFixedBlock)
  {
    NAM_FAIL(nam::Status::ErrorUnsupportedShape, "A2FastModel: maxBufferSize != NAM_A2_FIXED_BLOCK");
    return;
  }
  #endif

  DSP::SetMaxBufferSize(maxBufferSize);

  _layer_in.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _head_sum.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _z.assign(static_cast<size_t>(Channels) * maxBufferSize, 0.0f);
  _cond.assign(static_cast<size_t>(maxBufferSize), 0.0f);
  _head_out.assign(static_cast<size_t>(maxBufferSize), 0.0f);

  for (auto& L : _layers)
  {
  #if NAM_A2_RING_MODE == 1
    // Exactly max_lookback + one block, no rounding. The read window is then
    // precisely the whole ring: after writing this block's frames the deepest
    // tap sits at the position the NEXT block will overwrite, so nothing is
    // wasted and nothing is lost. Rounding up to a power of two, which is what
    // this used to do to make the wrap a mask, cost 40% (131.4K vs 78.9K of
    // history per model) and that 40% is the difference between two model
    // instances fitting in DTCM and not.
    L.ring_size = L.max_lookback + maxBufferSize;
    L.history.assign(static_cast<size_t>(Channels) * (L.ring_size + maxBufferSize), 0.0f);
    // Frame-major reads row c at [c * row_stride]; the reference kernel ignores
    // this. Same allocation, different interpretation.
    L.row_stride = L.ring_size + maxBufferSize;
    L.write_pos = L.max_lookback;
  #else
    L.history_cols = 2 * L.max_lookback + maxBufferSize;
    L.history.assign(static_cast<size_t>(Channels) * L.history_cols, 0.0f);
    L.write_pos = L.max_lookback;
  #endif
  }

  const int head_lookback = kHeadKernelSize - 1;
  #if NAM_A2_RING_MODE == 1
  _head_ring_size = head_lookback + maxBufferSize;
  _head_history.assign(static_cast<size_t>(Channels) * (_head_ring_size + maxBufferSize), 0.0f);
  _head_row_stride = _head_ring_size + maxBufferSize;
  _head_write_pos = head_lookback;
  #else
  _head_history_cols = 2 * head_lookback + maxBufferSize;
  _head_history.assign(static_cast<size_t>(Channels) * _head_history_cols, 0.0f);
  _head_write_pos = head_lookback;
  #endif
}

// -----------------------------------------------------------------------------
// Prewarm-state cache
//
// Processing silence for a full receptive field leaves every convolution
// history constant in time. Keep one Channels-wide column from each layer and
// the head so later prewarms can rebuild the complete histories directly.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::prewarm()
{
  if (HasCachedPrewarmState())
  {
    PrewarmFromCache();
    return;
  }

  DSP::prewarm();
  CacheStateAsPrewarmed();
}

template <int Channels>
void A2FastModel<Channels>::PrewarmFromCache()
{
  for (auto& L : _layers)
  {
    const size_t columns = L.history.size() / Channels;
    if (_frame_major())
    {
      // Row c is uniformly the cached value for channel c. Same steady state as
      // the column-major fill below, written in the transposed layout.
      for (int c = 0; c < Channels; c++)
      {
        float* row = L.history.data() + static_cast<size_t>(c) * L.row_stride;
        for (int t = 0; t < L.row_stride; t++)
          row[t] = L.cached_prewarm_state[c];
      }
    }
    else
    {
      for (size_t column = 0; column < columns; column++)
      {
        std::copy(L.cached_prewarm_state.begin(), L.cached_prewarm_state.end(),
                  L.history.begin() + static_cast<std::ptrdiff_t>(column * Channels));
      }
    }
    L.write_pos = L.max_lookback;
  }

  if (_frame_major())
  {
    for (int c = 0; c < Channels; c++)
    {
      float* row = _head_history.data() + static_cast<size_t>(c) * _head_row_stride;
      for (int t = 0; t < _head_row_stride; t++)
        row[t] = _cached_head_prewarm_state[c];
    }
  }
  else
  {
    const size_t head_columns = _head_history.size() / Channels;
    for (size_t column = 0; column < head_columns; column++)
    {
      std::copy(_cached_head_prewarm_state.begin(), _cached_head_prewarm_state.end(),
                _head_history.begin() + static_cast<std::ptrdiff_t>(column * Channels));
    }
  }
  _head_write_pos = kHeadKernelSize - 1;
}

template <int Channels>
void A2FastModel<Channels>::CacheStateAsPrewarmed()
{
  for (auto& L : _layers)
  {
  #if NAM_A2_RING_MODE == 1
    const int last_column = wrap_back(L.write_pos - 1, L.ring_size);
  #else
    const int last_column = L.write_pos - 1;
  #endif
    if (_frame_major())
    {
      for (int c = 0; c < Channels; c++)
        L.cached_prewarm_state[c] = L.history[static_cast<size_t>(c) * L.row_stride + last_column];
    }
    else
    {
      std::copy_n(L.history.begin() + static_cast<std::ptrdiff_t>(last_column * Channels), Channels,
                  L.cached_prewarm_state.begin());
    }
  }

  #if NAM_A2_RING_MODE == 1
  const int last_head_column = wrap_back(_head_write_pos - 1, _head_ring_size);
  #else
  const int last_head_column = _head_write_pos - 1;
  #endif
  if (_frame_major())
  {
    for (int c = 0; c < Channels; c++)
      _cached_head_prewarm_state[c] = _head_history[static_cast<size_t>(c) * _head_row_stride + last_head_column];
  }
  else
  {
    std::copy_n(_head_history.begin() + static_cast<std::ptrdiff_t>(last_head_column * Channels), Channels,
                _cached_head_prewarm_state.begin());
  }
  _has_cached_prewarm_state = true;
}

// -----------------------------------------------------------------------------
// Ring-write helpers.
//   Mode 1: pow2 + tail mirror. Constant-time per block (one short memcpy
//   into the ring, one mirror refresh).
//   Mode 0: linear with periodic memmove rewind. When write_pos nears the
//   end of history, memmove the trailing max_lookback cols back to offset 0
//   and reset write_pos. That memmove is the jitter spike we're measuring.
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_ring_write(Layer& L, int num_frames)
{
  #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames; // checked against kFixedBlock in process()
  #else
  const int nf = num_frames;
    #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
    #endif
  #endif

  #if NAM_A2_RING_MODE == 1
  float* const hist = L.history.data();
  const float* const src = _layer_in.data();
  const int wp = L.write_pos;
  // Split rather than std::min: on the no-wrap path the copy length is nf,
  // which is a compile-time constant under NAM_A2_FIXED_BLOCK, so GCC expands
  // it inline. Folded into a min() the length is runtime-variable and every
  // block pays an out-of-line newlib memcpy per layer. The wrap path keeps the
  // variable-length calls but only fires once per (ring_size / nf) blocks.
  const bool wrapped = (wp + nf > L.ring_size);
  if (!wrapped)
  {
    copy_block<Channels>(hist + static_cast<size_t>(wp) * Channels, src, nf);
  }
  else
  {
    const int first = L.ring_size - wp;
    std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(nf - first) * Channels * sizeof(float));
  }

  // The tail mirror only goes stale when this write actually touched columns
  // [0, mbs) -- either because it wrapped into them, or because it started
  // inside them. Refreshing it unconditionally, which is what this did, costs
  // a block-sized copy per layer per block to rewrite bytes that did not
  // change. For the deep layers (ring 1203, block 8) the mirror is live about
  // one block in 75; the other 74 were pure waste.
  if (wrapped || wp < mbs)
  {
    copy_block<Channels>(hist + static_cast<size_t>(L.ring_size) * Channels, hist, mbs);
  }
  L.write_pos = wrap_fwd(wp + nf, L.ring_size);
  #else
  if (L.write_pos + nf > L.history_cols)
  {
    const int keep = L.max_lookback;
    std::memmove(L.history.data(), L.history.data() + static_cast<size_t>(L.write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    L.write_pos = keep;
  }
  std::memcpy(L.history.data() + static_cast<size_t>(L.write_pos) * Channels, _layer_in.data(),
              static_cast<size_t>(nf) * Channels * sizeof(float));
  L.write_pos += nf;
  #endif
}

  #if NAM_A2_HAVE_MVE
// Frame-major ring write. Same pow2 + tail-mirror scheme as _ring_write, but
// applied per channel row: row c holds consecutive frames, so a tap read is a
// contiguous run and the mirror makes a wrapped read contiguous too.
//
// _layer_in is frame-major here as well, so the source for row c is its own
// contiguous run -- this is a set of small straight copies, not a scatter.
template <int Channels>
void A2FastModel<Channels>::_ring_write_fm(Layer& L, int num_frames)
{
    #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames;
    #else
  const int nf = num_frames;
  const int mbs = GetMaxBufferSize();
    #endif

  const int wp = L.write_pos;
  const int stride = L.row_stride;
  float* const hist = L.history.data();
  const float* const src = _layer_in.data();

  // Tail mirror: the first mbs frames repeated past ring_size, so a tap whose
  // window straddles the wrap still reads one contiguous run. Stale only when
  // this write touched [0, mbs) -- see the note in _ring_write.
  const bool wrapped = (wp + nf > L.ring_size);
  const bool mirror = wrapped || (wp < mbs);

  for (int c = 0; c < Channels; c++)
  {
    float* row = hist + static_cast<size_t>(c) * stride;
    const float* srow = src + static_cast<size_t>(c) * mbs;

    if (!wrapped)
    {
      copy_row(row + wp, srow, nf);
    }
    else
    {
      // memcpy, not element loops. `first` is a runtime length, and GCC gives
      // such a loop its fully versioned vectorizer treatment -- runtime trip
      // count, an alignment guard, and a scalar fallback -- which is a lot of
      // code for a path that runs about one block in 150. The column-major
      // _ring_write already used memcpy here; this matches it.
      const int first = L.ring_size - wp;
      std::memcpy(row + wp, srow, static_cast<size_t>(first) * sizeof(float));
      std::memcpy(row, srow + first, static_cast<size_t>(nf - first) * sizeof(float));
    }

    if (mirror)
    {
      copy_row(row + L.ring_size, row, mbs);
    }
  }

  L.write_pos = wrap_fwd(wp + nf, L.ring_size);
}
  #endif // NAM_A2_HAVE_MVE

template <int Channels>
void A2FastModel<Channels>::_head_ring_write(int num_frames)
{
  #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames; // checked against kFixedBlock in process()
  #else
  const int nf = num_frames;
    #if NAM_A2_RING_MODE == 1
  const int mbs = GetMaxBufferSize();
    #endif
  #endif

  #if NAM_A2_RING_MODE == 1
  float* const hist = _head_history.data();

  // Column-major head history, for the reference kernel only. The frame-major
  // kernels use _head_ring_write_fm, which needs no transpose: _head_sum is
  // already frame-major when they produce it, so the copy is row to row.
  const float* const src = _head_sum.data();

  const int wp = _head_write_pos;
  // Same no-wrap split as _ring_write, for the same reason.
  const bool wrapped = (wp + nf > _head_ring_size);
  if (!wrapped)
  {
    copy_block<Channels>(hist + static_cast<size_t>(wp) * Channels, src, nf);
  }
  else
  {
    const int first = _head_ring_size - wp;
    std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(nf - first) * Channels * sizeof(float));
  }

  // Conditional for the same reason as the layer mirror. Note the head mirror
  // is not actually read at present: _head_forward wraps every column index
  // individually through col_of(), so a head read never spans the wrap. It is
  // kept, rather than deleted, because vectorising the head means reading
  // contiguous runs of frames, and that does need it. The head ring is short
  // (23 columns to a block of 8), so this skips maybe 60% of the copies rather
  // than the ~99% the deep layers skip.
  if (wrapped || wp < mbs)
  {
    copy_block<Channels>(hist + static_cast<size_t>(_head_ring_size) * Channels, hist, mbs);
  }
  _head_write_pos = wrap_fwd(wp + nf, _head_ring_size);
  #else
  const int keep = kHeadKernelSize - 1;
  if (_head_write_pos + nf > _head_history_cols)
  {
    std::memmove(_head_history.data(), _head_history.data() + static_cast<size_t>(_head_write_pos - keep) * Channels,
                 static_cast<size_t>(keep) * Channels * sizeof(float));
    _head_write_pos = keep;
  }
  std::memcpy(_head_history.data() + static_cast<size_t>(_head_write_pos) * Channels, _head_sum.data(),
              static_cast<size_t>(nf) * Channels * sizeof(float));
  _head_write_pos += nf;
  #endif
}

// -----------------------------------------------------------------------------
// Per-layer forward pass. Reads current _layer_in, writes back into _layer_in
// after applying dilated conv + mixin + LeakyReLU + layer1x1 residual, and
// accumulates activations into _head_sum.
// -----------------------------------------------------------------------------
// Compile-time-specialized per-layer kernel. KernelSize is a template param
// so the K tap loop + per-tap weight offsets become compile-time constants;
// clang fully unrolls and can schedule FMAs across taps. Called from the
// runtime dispatcher below for each A2 kernel size (6 and 15).
template <int Channels>
template <int KernelSize>
void A2FastModel<Channels>::_layer_forward_k(Layer& L, const float* cond, int num_frames)
{
  constexpr int K = KernelSize;
  const int D = L.dilation;
  #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  (void)num_frames; // checked against kFixedBlock in process()
  #else
  const int nf = num_frames;
  #endif
  // Physical ring position of this block's first frame, offset by `taps_back *
  // D` samples into the past. In pow2 mode the position is wrapped by mask and
  // reads spanning the wrap land in the tail mirror; in linear mode write_pos
  // is monotonic and arithmetic is plain.
  #if NAM_A2_RING_MODE == 1
  const int rsz = L.ring_size;
  auto tap_base_phys = [&](int taps_back) { return wrap_back(L.write_pos - nf - taps_back * D, rsz); };
  #else
  const int base = L.write_pos - nf;
  auto tap_base_phys = [&](int taps_back) { return base - taps_back * D; };
  #endif

  // Two conv strategies, dispatched at compile time on Channels:
  //
  //   - Channels <= 4 (A2-Lite): full-block tap-major. The z accumulator lives
  //     in the heap buffer across all taps, and for each tap the inner f-loop
  //     iterates over all nf. This gives clang frame-level
  //     parallelism — it vectorizes across 4 frames at a time, which matters
  //     more than weight-reload cost when the b-loop (3 wide) can't saturate
  //     NEON lanes on its own.
  //
  //   - Channels >= 8 (A2-Full): frame-tiled tap-major with T=4. ztile
  //     stays in NEON registers across all K taps, amortizing weight loads
  //     over 4 frames — equivalent to what a GEMM kernel does. Weight reuse
  //     matters here because the b-loop (8 wide) already saturates SIMD, so
  //     frame-level parallelism gives no extra headroom. The 1x1 residual is
  //     also tiled over the same T=4 frames so W1x1 loads are amortized.

  if constexpr (Channels == 3)
  {
    // Inner 3x3 GEMV fully unrolled: all 9 weights lifted into named consts
    // before the frame loop, the c-reduction kept in scalar temps a0/a1/a2 so
    // the compiler keeps them in FP registers across the frame loop. Mirrors
    // the nam2c --fused structure.
    float* z = _z.data();

    // Tap 0: seed z with conv_b (saves the memset-to-zero pass) and fold in
    // the first tap's FMAs.
    {
      const float* wk = &L.conv_w[0];
      const int tap_base = tap_base_phys(K - 1);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      const float cb0 = L.conv_b[0], cb1 = L.conv_b[1], cb2 = L.conv_b[2];
      for (int f = 0; f < nf; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float a0 = cb0 + w0 * src[0];
        float a1 = cb1 + w1 * src[0];
        float a2 = cb2 + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        float* zf = z + static_cast<size_t>(f) * 3;
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Taps 1..K-2: accumulate into z with the same unrolled inner kernel.
    for (int k = 1; k < K - 1; k++)
    {
      const float* wk = &L.conv_w[static_cast<size_t>(k) * 9];
      const int tap_base = tap_base_phys(K - 1 - k);
      const float w0 = wk[0], w1 = wk[1], w2 = wk[2];
      const float w3 = wk[3], w4 = wk[4], w5 = wk[5];
      const float w6 = wk[6], w7 = wk[7], w8 = wk[8];
      for (int f = 0; f < nf; f++)
      {
        const float* src = &L.history[static_cast<size_t>(tap_base + f) * 3];
        float* zf = z + static_cast<size_t>(f) * 3;
        float a0 = zf[0] + w0 * src[0];
        float a1 = zf[1] + w1 * src[0];
        float a2 = zf[2] + w2 * src[0];
        a0 += w3 * src[1];
        a1 += w4 * src[1];
        a2 += w5 * src[1];
        a0 += w6 * src[2];
        a1 += w7 * src[2];
        a2 += w8 * src[2];
        zf[0] = a0;
        zf[1] = a1;
        zf[2] = a2;
      }
    }

    // Final tap (K-1, offset 0) fully inlined with the post-conv tail.
    // Everything runs on register-resident scalars:
    //   conv tap K-1 -> mixin -> LeakyReLU -> head_sum += -> layer1x1 residual.
    const float* wk_last = &L.conv_w[static_cast<size_t>(K - 1) * 9];
    const int tap_base_last = tap_base_phys(0);
    const float cw0 = wk_last[0], cw1 = wk_last[1], cw2 = wk_last[2];
    const float cw3 = wk_last[3], cw4 = wk_last[4], cw5 = wk_last[5];
    const float cw6 = wk_last[6], cw7 = wk_last[7], cw8 = wk_last[8];
    const float mw0 = L.mixin_w[0], mw1 = L.mixin_w[1], mw2 = L.mixin_w[2];
    // layer1x1 col-major: lw[b*3 + c] is weight from bottleneck b to output c.
    const float lw00 = L.l1x1_w[0], lw01 = L.l1x1_w[1], lw02 = L.l1x1_w[2];
    const float lw10 = L.l1x1_w[3], lw11 = L.l1x1_w[4], lw12 = L.l1x1_w[5];
    const float lw20 = L.l1x1_w[6], lw21 = L.l1x1_w[7], lw22 = L.l1x1_w[8];
    const float lb0 = L.l1x1_b[0], lb1 = L.l1x1_b[1], lb2 = L.l1x1_b[2];
    for (int f = 0; f < nf; f++)
    {
      const float* src = &L.history[static_cast<size_t>(tap_base_last + f) * 3];
      const float* zf_mem = z + static_cast<size_t>(f) * 3;
      // Final tap GEMV.
      float a0 = zf_mem[0] + cw0 * src[0];
      float a1 = zf_mem[1] + cw1 * src[0];
      float a2 = zf_mem[2] + cw2 * src[0];
      a0 += cw3 * src[1];
      a1 += cw4 * src[1];
      a2 += cw5 * src[1];
      a0 += cw6 * src[2];
      a1 += cw7 * src[2];
      a2 += cw8 * src[2];
      // Mixin + LeakyReLU.
      const float cf = cond[f];
      a0 += mw0 * cf;
      a1 += mw1 * cf;
      a2 += mw2 * cf;
      // The ternary below compiles to VCMPE.F32 -> VMRS APSR_nzcv, fpscr -> IT/VMULMI, and
      // that VMRS drains the FP pipeline (552 of them per block here). A
      // branchless max(a,0) + slope*min(a,0) form removes them entirely --
      // measured, it lowered to VMAXNM/VMINNM with no flag traffic as intended,
      // and made the block *slightly slower*. The loop is not issue-limited on
      // this path, so there is nothing to win here; keep the plain form, which
      // also preserves the sign of zero exactly as the reference does.
      a0 = (a0 >= 0.0f) ? a0 : a0 * kLeakySlope;
      a1 = (a1 >= 0.0f) ? a1 : a1 * kLeakySlope;
      a2 = (a2 >= 0.0f) ? a2 : a2 * kLeakySlope;
      // Head sum accumulate.
      float* hsum = &_head_sum[static_cast<size_t>(f) * 3];
      hsum[0] += a0;
      hsum[1] += a1;
      hsum[2] += a2;
      // layer1x1 residual.
      float* lin = &_layer_in[static_cast<size_t>(f) * 3];
      lin[0] += lb0 + lw00 * a0 + lw10 * a1 + lw20 * a2;
      lin[1] += lb1 + lw01 * a0 + lw11 * a1 + lw21 * a2;
      lin[2] += lb2 + lw02 * a0 + lw12 * a1 + lw22 * a2;
    }
  }
  else
  {
    // Use Eigen's tuned 8x8 × 8xN GEMM for the whole block at once. Unlike a
    // small-tile version, this hits Eigen's actual GEMM kernel (tuned for
    // inner dimensions of ~64) rather than its tiny-matrix fallback path.
    //
    // Compile-time improvements over the generic WaveNet path:
    //   - Channels and Bottleneck are template constants (no dynamic shape).
    //   - Per-layer buffers are pre-sized at SetMaxBufferSize; nothing resizes
    //     during process().
    //   - No FiLM / gating / head1x1 / grouped-conv branches.
    //   - No virtual dispatch / conditional on optional layer features.
    //   - All conv + post-conv ops operate on the full block — even the
    //     mixin, bias, activation, and 1x1 residual are Eigen block ops so
    //     they vectorize the same way the GEMMs do.
    using MatCC = Eigen::Matrix<float, Channels, Channels>;
    using MatCDyn = Eigen::Matrix<float, Channels, Eigen::Dynamic>;
    using VecC = Eigen::Matrix<float, Channels, 1>;
    using RowDyn = Eigen::Matrix<float, 1, Eigen::Dynamic>;

    Eigen::Map<const VecC> conv_b_vec(L.conv_b.data());
    Eigen::Map<const VecC> mixin_vec(L.mixin_w.data());
    Eigen::Map<const MatCC> l1x1_mat(L.l1x1_w.data());
    Eigen::Map<const VecC> l1x1_b_vec(L.l1x1_b.data());
    Eigen::Map<const RowDyn> cond_row(cond, 1, nf);

    Eigen::Map<MatCDyn> ztile(_z.data(), Channels, nf);
    Eigen::Map<MatCDyn> hsum_block(_head_sum.data(), Channels, nf);
    Eigen::Map<MatCDyn> lin_block(_layer_in.data(), Channels, nf);

    ztile.setZero();

    // Conv: one 8x8 × 8xN GEMM per tap.
    for (int k = 0; k < K; k++)
    {
      const int tap_base = tap_base_phys(K - 1 - k);
      Eigen::Map<const MatCC> W(&L.conv_w[static_cast<size_t>(k) * Channels * Channels]);
      Eigen::Map<const MatCDyn> input_block(&L.history[static_cast<size_t>(tap_base) * Channels], Channels, nf);
      ztile.noalias() += W * input_block;
    }

    // Post-conv: bias, mixin, LeakyReLU, head_sum, 1x1 residual — all block ops.
    ztile.colwise() += conv_b_vec;
    ztile.noalias() += mixin_vec * cond_row; // rank-1 outer product
    ztile = (ztile.array() < 0.0f).select(ztile.array() * kLeakySlope, ztile.array());
    hsum_block += ztile;
    lin_block.noalias() += l1x1_mat * ztile; // 8x8 × 8xN GEMM
    lin_block.colwise() += l1x1_b_vec;
  }
}

  #if NAM_A2_HAVE_MVE
// -----------------------------------------------------------------------------
// Frame-major Helium kernel (Channels == 3).
//
// Layout is the whole point. With the history transposed so that row c holds
// consecutive frames of channel c, one vldrw.32 fetches four frames of one
// channel, and the 3x3 GEMV becomes vector-by-scalar FMAs:
//
//     z_i[f..f+3] += w[j*3 + i] * x_j[f..f+3]
//
// where x_j is a vector load and the weight is a scalar in a general register.
// Nine VFMAs cover four frames of the full 3x3, against 36 scalar FMAs in the
// reference kernel -- and the three accumulators stay in Q registers across all
// K taps instead of round-tripping through _z.
//
// Bit-exactness: per lane, the accumulation order is identical to the reference
// kernel (seed with conv_b, then tap 0 j=0,1,2, tap 1 j=0,1,2, ...). Lanes are
// independent, so this should agree to the bit, not merely to a tolerance. The
// on-target harness checks exactly that.
//
// nf is a multiple of 4 (asserted below), processed as nf/4 vector groups.
// -----------------------------------------------------------------------------
template <int Channels>
template <int KernelSize>
void A2FastModel<Channels>::_layer_forward_fm(Layer& L, const float* cond, int num_frames)
{
  static_assert(Channels == 3, "frame-major kernel is written for a 3-channel layer");
  constexpr int K = KernelSize;

    #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames;
  static_assert(nf % 4 == 0, "frame-major kernel processes 4 frames per vector");
    #else
  const int nf = num_frames;
  const int mbs = GetMaxBufferSize();
    #endif

  // _layer_in / _head_sum are Channels rows of max_buffer_size, so the row
  // stride is mbs -- not nf, which is only equal to it under a fixed block.
  const int rs = mbs;

  const int D = L.dilation;
  const int rsz = L.ring_size;
  const int stride = L.row_stride;
  const float* const hist = L.history.data();

  // Physical frame index of this block's first frame, taps_back*D in the past.
  // The tail mirror guarantees [base, base + nf) is contiguous within a row.
  auto tap_base = [&](int taps_back) { return wrap_back(L.write_pos - nf - taps_back * D, rsz); };

  float* const lin = _layer_in.data();
  float* const hsum = _head_sum.data();

  const float* const cw = L.conv_w.data();
  const float* const lw = L.l1x1_w.data();

  for (int g = 0; g < nf; g += 4)
  {
    // Accumulators for this group of four frames, one per output channel,
    // resident in Q registers for the whole tap loop.
    float32x4_t z0 = vdupq_n_f32(L.conv_b[0]);
    float32x4_t z1 = vdupq_n_f32(L.conv_b[1]);
    float32x4_t z2 = vdupq_n_f32(L.conv_b[2]);

    for (int k = 0; k < K; k++)
    {
      const float* w = cw + static_cast<size_t>(k) * 9;
      const int b = tap_base(K - 1 - k) + g;
      const float32x4_t x0 = vld1q_f32(hist + b);
      const float32x4_t x1 = vld1q_f32(hist + stride + b);
      const float32x4_t x2 = vld1q_f32(hist + 2 * stride + b);

      // w[j*3 + i] is input j -> output i, matching the reference kernel's
      // w0..w8 ordering exactly.
      z0 = vfmaq_n_f32(z0, x0, w[0]);
      z1 = vfmaq_n_f32(z1, x0, w[1]);
      z2 = vfmaq_n_f32(z2, x0, w[2]);
      z0 = vfmaq_n_f32(z0, x1, w[3]);
      z1 = vfmaq_n_f32(z1, x1, w[4]);
      z2 = vfmaq_n_f32(z2, x1, w[5]);
      z0 = vfmaq_n_f32(z0, x2, w[6]);
      z1 = vfmaq_n_f32(z1, x2, w[7]);
      z2 = vfmaq_n_f32(z2, x2, w[8]);
    }

    // Mixin.
    const float32x4_t cf = vld1q_f32(cond + g);
    z0 = vfmaq_n_f32(z0, cf, L.mixin_w[0]);
    z1 = vfmaq_n_f32(z1, cf, L.mixin_w[1]);
    z2 = vfmaq_n_f32(z2, cf, L.mixin_w[2]);

    // LeakyReLU as a predicated multiply: lanes below zero get scaled, the rest
    // are left alone. Two instructions and no flag traffic, unlike the scalar
    // path's VCMPE/VMRS pair -- which is free to be slow there because it is a
    // third of the work here.
    z0 = vmulq_m_n_f32(z0, z0, kLeakySlope, vcmpltq_n_f32(z0, 0.0f));
    z1 = vmulq_m_n_f32(z1, z1, kLeakySlope, vcmpltq_n_f32(z1, 0.0f));
    z2 = vmulq_m_n_f32(z2, z2, kLeakySlope, vcmpltq_n_f32(z2, 0.0f));

    // Head-sum accumulate, frame-major rows.
    vst1q_f32(hsum + g, vaddq_f32(vld1q_f32(hsum + g), z0));
    vst1q_f32(hsum + rs + g, vaddq_f32(vld1q_f32(hsum + rs + g), z1));
    vst1q_f32(hsum + 2 * rs + g, vaddq_f32(vld1q_f32(hsum + 2 * rs + g), z2));

    // layer1x1 residual: lin_i += l1x1_b[i] + sum_b lw[b*3 + i] * z_b.
    //
    // Grouped exactly as the reference writes it -- the bias-seeded sum is
    // formed first and added to lin as one step. Hoisting the load of lin into
    // the accumulator instead would be algebraically the same but a different
    // rounding sequence, and would cost the bit-exact comparison.
    float32x4_t t0 = vdupq_n_f32(L.l1x1_b[0]);
    float32x4_t t1 = vdupq_n_f32(L.l1x1_b[1]);
    float32x4_t t2 = vdupq_n_f32(L.l1x1_b[2]);
    t0 = vfmaq_n_f32(t0, z0, lw[0]);
    t1 = vfmaq_n_f32(t1, z0, lw[1]);
    t2 = vfmaq_n_f32(t2, z0, lw[2]);
    t0 = vfmaq_n_f32(t0, z1, lw[3]);
    t1 = vfmaq_n_f32(t1, z1, lw[4]);
    t2 = vfmaq_n_f32(t2, z1, lw[5]);
    t0 = vfmaq_n_f32(t0, z2, lw[6]);
    t1 = vfmaq_n_f32(t1, z2, lw[7]);
    t2 = vfmaq_n_f32(t2, z2, lw[8]);
    vst1q_f32(lin + g, vaddq_f32(vld1q_f32(lin + g), t0));
    vst1q_f32(lin + rs + g, vaddq_f32(vld1q_f32(lin + rs + g), t1));
    vst1q_f32(lin + 2 * rs + g, vaddq_f32(vld1q_f32(lin + 2 * rs + g), t2));
  }
}
// -----------------------------------------------------------------------------
// Frame-major Helium kernel, both frame groups carried through the tap loop.
//
// Same math and same layout as _layer_forward_fm; only the loop nest differs.
// That kernel wraps the tap loop inside the group loop, so every weight is
// re-loaded for each group of four frames -- the generated code is a strict
// alternation of `ldr` and `vfma.f32 q`, one scalar load per vector FMA, and
// the tap's address arithmetic (mask, scale, three row offsets) is likewise
// paid twice.
//
// Here both groups live across the tap loop, so a weight is loaded once and
// feeds two FMAs, and the tap addressing is computed once. The tap base also
// advances by +D per tap rather than being recomputed with a multiply.
//
// Register budget is exactly the 8 Q registers MVE has: six accumulators
// (3 channels x 2 groups) plus two in-flight input vectors, which is why the
// inputs are consumed one channel at a time rather than all three being held.
// If this spills it will show up as *worse* than fm, which is precisely the
// kind of thing the harness is for.
// -----------------------------------------------------------------------------
template <int Channels>
template <int KernelSize>
void A2FastModel<Channels>::_layer_forward_fm2(Layer& L, const float* cond, int num_frames)
{
  static_assert(Channels == 3, "frame-major kernel is written for a 3-channel layer");
  constexpr int K = KernelSize;

    #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames;
  static_assert(nf == 8, "fm2 carries exactly two 4-frame groups");
    #else
  const int nf = num_frames;
  const int mbs = GetMaxBufferSize();
    #endif

  const int rs = mbs;
  const int D = L.dilation;
  const int rsz = L.ring_size;
  const int stride = L.row_stride;

  // Row bases hoisted out of the tap loop: they are loop-invariant, but the
  // compiler recomputes hist + n*stride + b per tap when they are not.
  const float* const row0 = L.history.data();
  const float* const row1 = row0 + stride;
  const float* const row2 = row1 + stride;

  float* const lin = _layer_in.data();
  float* const hsum = _head_sum.data();
  const float* const cw = L.conv_w.data();
  const float* const lw = L.l1x1_w.data();

  float32x4_t z0a = vdupq_n_f32(L.conv_b[0]);
  float32x4_t z0b = z0a;
  float32x4_t z1a = vdupq_n_f32(L.conv_b[1]);
  float32x4_t z1b = z1a;
  float32x4_t z2a = vdupq_n_f32(L.conv_b[2]);
  float32x4_t z2b = z2a;

  // Deepest tap first, then +D per tap: the same sequence tap_base(K-1-k)
  // produces, without the per-tap multiply.
  // With the ring sized exactly max_lookback + nf, the deepest tap base is
  // (write_pos - nf - max_lookback) mod ring_size, and nf + max_lookback IS
  // ring_size -- so it reduces to write_pos itself, with no arithmetic at all.
  int b = L.write_pos;

  for (int k = 0; k < K; k++)
  {
    const float* w = cw + static_cast<size_t>(k) * 9;

    // One input vector live at a time: six accumulators plus one input is seven
    // Q registers, leaving one spare. Holding both groups' inputs at once needs
    // all eight and GCC spills, which costs more than the reload it saves. The
    // three weights of each row still serve both groups from GP registers.
    float32x4_t x = vld1q_f32(row0 + b);
    z0a = vfmaq_n_f32(z0a, x, w[0]);
    z1a = vfmaq_n_f32(z1a, x, w[1]);
    z2a = vfmaq_n_f32(z2a, x, w[2]);
    x = vld1q_f32(row0 + b + 4);
    z0b = vfmaq_n_f32(z0b, x, w[0]);
    z1b = vfmaq_n_f32(z1b, x, w[1]);
    z2b = vfmaq_n_f32(z2b, x, w[2]);

    x = vld1q_f32(row1 + b);
    z0a = vfmaq_n_f32(z0a, x, w[3]);
    z1a = vfmaq_n_f32(z1a, x, w[4]);
    z2a = vfmaq_n_f32(z2a, x, w[5]);
    x = vld1q_f32(row1 + b + 4);
    z0b = vfmaq_n_f32(z0b, x, w[3]);
    z1b = vfmaq_n_f32(z1b, x, w[4]);
    z2b = vfmaq_n_f32(z2b, x, w[5]);

    x = vld1q_f32(row2 + b);
    z0a = vfmaq_n_f32(z0a, x, w[6]);
    z1a = vfmaq_n_f32(z1a, x, w[7]);
    z2a = vfmaq_n_f32(z2a, x, w[8]);
    x = vld1q_f32(row2 + b + 4);
    z0b = vfmaq_n_f32(z0b, x, w[6]);
    z1b = vfmaq_n_f32(z1b, x, w[7]);
    z2b = vfmaq_n_f32(z2b, x, w[8]);

    b = wrap_fwd(b + D, rsz);
  }

  // Post-conv tail, identical to fm but for both groups.
  const float32x4_t ca = vld1q_f32(cond);
  const float32x4_t cb = vld1q_f32(cond + 4);
  z0a = vfmaq_n_f32(z0a, ca, L.mixin_w[0]);
  z0b = vfmaq_n_f32(z0b, cb, L.mixin_w[0]);
  z1a = vfmaq_n_f32(z1a, ca, L.mixin_w[1]);
  z1b = vfmaq_n_f32(z1b, cb, L.mixin_w[1]);
  z2a = vfmaq_n_f32(z2a, ca, L.mixin_w[2]);
  z2b = vfmaq_n_f32(z2b, cb, L.mixin_w[2]);

  z0a = vmulq_m_n_f32(z0a, z0a, kLeakySlope, vcmpltq_n_f32(z0a, 0.0f));
  z0b = vmulq_m_n_f32(z0b, z0b, kLeakySlope, vcmpltq_n_f32(z0b, 0.0f));
  z1a = vmulq_m_n_f32(z1a, z1a, kLeakySlope, vcmpltq_n_f32(z1a, 0.0f));
  z1b = vmulq_m_n_f32(z1b, z1b, kLeakySlope, vcmpltq_n_f32(z1b, 0.0f));
  z2a = vmulq_m_n_f32(z2a, z2a, kLeakySlope, vcmpltq_n_f32(z2a, 0.0f));
  z2b = vmulq_m_n_f32(z2b, z2b, kLeakySlope, vcmpltq_n_f32(z2b, 0.0f));

  vst1q_f32(hsum, vaddq_f32(vld1q_f32(hsum), z0a));
  vst1q_f32(hsum + 4, vaddq_f32(vld1q_f32(hsum + 4), z0b));
  vst1q_f32(hsum + rs, vaddq_f32(vld1q_f32(hsum + rs), z1a));
  vst1q_f32(hsum + rs + 4, vaddq_f32(vld1q_f32(hsum + rs + 4), z1b));
  vst1q_f32(hsum + 2 * rs, vaddq_f32(vld1q_f32(hsum + 2 * rs), z2a));
  vst1q_f32(hsum + 2 * rs + 4, vaddq_f32(vld1q_f32(hsum + 2 * rs + 4), z2b));

  float32x4_t t0a = vdupq_n_f32(L.l1x1_b[0]);
  float32x4_t t0b = t0a;
  float32x4_t t1a = vdupq_n_f32(L.l1x1_b[1]);
  float32x4_t t1b = t1a;
  float32x4_t t2a = vdupq_n_f32(L.l1x1_b[2]);
  float32x4_t t2b = t2a;
  t0a = vfmaq_n_f32(t0a, z0a, lw[0]);
  t0b = vfmaq_n_f32(t0b, z0b, lw[0]);
  t1a = vfmaq_n_f32(t1a, z0a, lw[1]);
  t1b = vfmaq_n_f32(t1b, z0b, lw[1]);
  t2a = vfmaq_n_f32(t2a, z0a, lw[2]);
  t2b = vfmaq_n_f32(t2b, z0b, lw[2]);
  t0a = vfmaq_n_f32(t0a, z1a, lw[3]);
  t0b = vfmaq_n_f32(t0b, z1b, lw[3]);
  t1a = vfmaq_n_f32(t1a, z1a, lw[4]);
  t1b = vfmaq_n_f32(t1b, z1b, lw[4]);
  t2a = vfmaq_n_f32(t2a, z1a, lw[5]);
  t2b = vfmaq_n_f32(t2b, z1b, lw[5]);
  t0a = vfmaq_n_f32(t0a, z2a, lw[6]);
  t0b = vfmaq_n_f32(t0b, z2b, lw[6]);
  t1a = vfmaq_n_f32(t1a, z2a, lw[7]);
  t1b = vfmaq_n_f32(t1b, z2b, lw[7]);
  t2a = vfmaq_n_f32(t2a, z2a, lw[8]);
  t2b = vfmaq_n_f32(t2b, z2b, lw[8]);

  vst1q_f32(lin, vaddq_f32(vld1q_f32(lin), t0a));
  vst1q_f32(lin + 4, vaddq_f32(vld1q_f32(lin + 4), t0b));
  vst1q_f32(lin + rs, vaddq_f32(vld1q_f32(lin + rs), t1a));
  vst1q_f32(lin + rs + 4, vaddq_f32(vld1q_f32(lin + rs + 4), t1b));
  vst1q_f32(lin + 2 * rs, vaddq_f32(vld1q_f32(lin + 2 * rs), t2a));
  vst1q_f32(lin + 2 * rs + 4, vaddq_f32(vld1q_f32(lin + 2 * rs + 4), t2b));
}
  #endif // NAM_A2_HAVE_MVE

// Runtime dispatcher: selects the K-specialized kernel for this layer.
// For the A2 shape the detector only admits K in {6, 15}; any other value
// here means something passed the detector that shouldn't have.
template <int Channels>
void A2FastModel<Channels>::_layer_forward(int layer_idx, const float* cond, int num_frames)
{
  Layer& L = _layers[layer_idx];

  #if NAM_A2_HAVE_MVE
  if constexpr (Channels == 3)
  {
    if (_frame_major())
    {
      _ring_write_fm(L, num_frames);
      NAM_A2_PROF_ENTER(kNamA2ProbeLayerMath);
      if (_kernel == 2)
      {
        switch (L.kernel_size)
        {
          case 6: _layer_forward_fm2<6>(L, cond, num_frames); break;
          case 15: _layer_forward_fm2<15>(L, cond, num_frames); break;
          default: NAM_FAIL(nam::Status::ErrorUnsupportedShape, "A2FastModel: unexpected kernel_size"); break;
        }
      }
      else
      {
        switch (L.kernel_size)
        {
          case 6: _layer_forward_fm<6>(L, cond, num_frames); break;
          case 15: _layer_forward_fm<15>(L, cond, num_frames); break;
          default: NAM_FAIL(nam::Status::ErrorUnsupportedShape, "A2FastModel: unexpected kernel_size"); break;
        }
      }
      NAM_A2_PROF_EXIT(kNamA2ProbeLayerMath);
      return;
    }
  }
  #endif

  _ring_write(L, num_frames);
  NAM_A2_PROF_ENTER(kNamA2ProbeLayerMath);
  switch (L.kernel_size)
  {
    case 6: _layer_forward_k<6>(L, cond, num_frames); break;
    case 15: _layer_forward_k<15>(L, cond, num_frames); break;
    // Unreachable: is_a2_shape() admits only K in {6, 15}. Latch rather than
    // throw so the audio path stays exception-free; the block is left silent.
    default: NAM_FAIL(nam::Status::ErrorUnsupportedShape, "A2FastModel: unexpected kernel_size"); break;
  }
  NAM_A2_PROF_EXIT(kNamA2ProbeLayerMath);
}

// -----------------------------------------------------------------------------
// Head: K=16 dilation-1 conv from Channels to 1, plus bias + scale.
// -----------------------------------------------------------------------------
  #if NAM_A2_HAVE_MVE
// Frame-major head ring write. The layer version, applied to the head's own
// (much shorter) ring: 23 columns to a block of 8.
template <int Channels>
void A2FastModel<Channels>::_head_ring_write_fm(int num_frames)
{
    #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  constexpr int mbs = kFixedBlock;
  (void)num_frames;
    #else
  const int nf = num_frames;
  const int mbs = GetMaxBufferSize();
    #endif

  const int wp = _head_write_pos;
  const int stride = _head_row_stride;
  float* const hist = _head_history.data();
  const float* const src = _head_sum.data();

  const bool wrapped = (wp + nf > _head_ring_size);
  // Unlike the old scalar head, the mirror is now genuinely load-bearing:
  // _head_forward_fm reads four consecutive frames of a row at a time, so a
  // window straddling the wrap lands in it.
  const bool mirror = wrapped || (wp < mbs);

  for (int c = 0; c < Channels; c++)
  {
    float* row = hist + static_cast<size_t>(c) * stride;
    const float* srow = src + static_cast<size_t>(c) * mbs;

    if (!wrapped)
    {
      copy_row(row + wp, srow, nf);
    }
    else
    {
      const int first = _head_ring_size - wp;
      std::memcpy(row + wp, srow, static_cast<size_t>(first) * sizeof(float));
      std::memcpy(row, srow + first, static_cast<size_t>(nf - first) * sizeof(float));
    }

    if (mirror)
    {
      copy_row(row + _head_ring_size, row, mbs);
    }
  }

  _head_write_pos = wrap_fwd(wp + nf, _head_ring_size);
}

// -----------------------------------------------------------------------------
// Frame-major Helium head.
//
// The head is a K=16, dilation-1 conv from Channels to 1. Scalar and
// column-major it was 16 taps x 3 channels x 8 frames = 384 FMAs with a load
// each -- the last unvectorised arithmetic in the block.
//
// Transposed, the columns a given tap reads for four consecutive frames are
// themselves consecutive, so each (tap, channel) is one vector load and one
// vector-by-scalar FMA: 96 vector FMAs in place of 384 scalar ones. The
// accumulator stays in a Q register across all 16 taps, and the per-block
// _head_sum transpose that the column-major layout forced disappears with it.
//
// Per lane the accumulation order matches the scalar head exactly -- seed with
// _head_b, then tap 0 channels 0,1,2, tap 1 channels 0,1,2, ...
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::_head_forward_fm(float* output, int num_frames)
{
  static_assert(Channels == 3, "frame-major head is written for a 3-channel layer");

  _head_ring_write_fm(num_frames);

    #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  (void)num_frames;
  static_assert(nf % 4 == 0, "frame-major head processes 4 frames per vector");
    #else
  const int nf = num_frames;
    #endif

  const int rsz = _head_ring_size;
  const int stride = _head_row_stride;
  const float* const row0 = _head_history.data();
  const float* const row1 = row0 + stride;
  const float* const row2 = row1 + stride;

  // First frame of this block, in ring coordinates. Can be negative; every
  // column index below is wrapped once, which is enough because the most
  // negative value it reaches is exactly -rsz.
  const int base = _head_write_pos - nf;

  for (int g = 0; g < nf; g += 4)
  {
    float32x4_t y = vdupq_n_f32(_head_b);

    for (int k = 0; k < kHeadKernelSize; k++)
    {
      // For a fixed tap, frames g..g+3 read columns c0..c0+3 -- contiguous,
      // which is the whole reason for the transpose.
      const int c0 = wrap_back(base - (kHeadKernelSize - 1 - k) + g, rsz);
      const float* const w = _head_w[k].data();
      y = vfmaq_n_f32(y, vld1q_f32(row0 + c0), w[0]);
      y = vfmaq_n_f32(y, vld1q_f32(row1 + c0), w[1]);
      y = vfmaq_n_f32(y, vld1q_f32(row2 + c0), w[2]);
    }

    vst1q_f32(output + g, vmulq_n_f32(y, _head_scale));
  }
}
  #endif // NAM_A2_HAVE_MVE

template <int Channels>
void A2FastModel<Channels>::_head_forward(float* output, int num_frames)
{
  #if NAM_A2_HAVE_MVE
  if constexpr (Channels == 3)
  {
    if (_frame_major())
    {
      _head_forward_fm(output, num_frames);
      return;
    }
  }
  #endif

  _head_ring_write(num_frames);
  #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  (void)num_frames; // checked against kFixedBlock in process()
  #else
  const int nf = num_frames;
  #endif
  #if NAM_A2_RING_MODE == 1
  const int rsz = _head_ring_size;
  auto col_of = [&](int f, int k) { return wrap_back(_head_write_pos - nf + f - (kHeadKernelSize - 1 - k), rsz); };
  #else
  const int base = _head_write_pos - nf;
  auto col_of = [&](int f, int k) { return base + f - (kHeadKernelSize - 1 - k); };
  #endif

  for (int f = 0; f < nf; f++)
  {
    float y = _head_b;
    for (int k = 0; k < kHeadKernelSize; k++)
    {
      const int col = col_of(f, k);
      const float* src = &_head_history[static_cast<size_t>(col) * Channels];
      const float* wk = _head_w[k].data();
      for (int b = 0; b < Channels; b++)
        y += wk[b] * src[b];
    }
    output[f] = y * _head_scale;
  }
}

// -----------------------------------------------------------------------------
// DSP::process override
// -----------------------------------------------------------------------------
template <int Channels>
void A2FastModel<Channels>::process(NAM_SAMPLE** input, NAM_SAMPLE** output, int num_frames)
{
  #if NAM_A2_FIXED_BLOCK > 0
  // Every bound in the hot path below is compiled for exactly kFixedBlock
  // frames, so a caller asking for a different count would read and write past
  // them. Refuse the block instead: latch, and leave the output untouched.
  // SetMaxBufferSize() rejects a mismatched size at Reset() time, so reaching
  // here means the caller changed its block size mid-stream.
  if (num_frames != kFixedBlock)
  {
    NAM_FAIL(nam::Status::ErrorUnsupportedShape, "A2FastModel: num_frames != NAM_A2_FIXED_BLOCK");
    return;
  }
  constexpr int nf = kFixedBlock;
  #else
  if (num_frames > GetMaxBufferSize())
    SetMaxBufferSize(num_frames);
  const int nf = num_frames;
  #endif

  const NAM_SAMPLE* in0 = input[0];
  NAM_SAMPLE* out0 = output[0];

  // Rechannel: layer_in[c, f] = _rechannel_w[c] * input[f] for c in Channels.
  // Also prepare float cond buffer (input copied to float for inner loops).
  float* cond = _cond.data();
  if (_frame_major())
  {
    // _layer_in laid out as Channels rows of max_buffer_size frames.
    const int rs = GetMaxBufferSize();
    for (int f = 0; f < nf; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      for (int c = 0; c < Channels; c++)
        _layer_in[static_cast<size_t>(c) * rs + f] = _rechannel_w[c] * x;
    }
  }
  else
  {
    for (int f = 0; f < nf; f++)
    {
      const float x = static_cast<float>(in0[f]);
      cond[f] = x;
      float* lin = &_layer_in[static_cast<size_t>(f) * Channels];
      for (int c = 0; c < Channels; c++)
        lin[c] = _rechannel_w[c] * x;
    }
  }

  // Zero head accumulator.
  std::memset(_head_sum.data(), 0, static_cast<size_t>(nf) * Channels * sizeof(float));

  for (int li = 0; li < kNumLayers; li++)
    _layer_forward(li, cond, nf);

  // Output.
  float* head_out = _head_out.data();
  _head_forward(head_out, nf);
  for (int f = 0; f < nf; f++)
    out0[f] = static_cast<NAM_SAMPLE>(head_out[f]);
}

// -----------------------------------------------------------------------------
// A2FastConfig — wraps the constructed DSP behind the ModelConfig interface.
// -----------------------------------------------------------------------------
struct A2FastConfig : public ModelConfig
{
  int channels = 0;

  std::unique_ptr<DSP> create(std::vector<float> weights, double sampleRate) override
  {
    std::unique_ptr<DSP> out;
    if (channels == 3)
      out = std::make_unique<A2FastModel<3>>(std::move(weights), sampleRate);
    else if (channels == 8)
      out = std::make_unique<A2FastModel<8>>(std::move(weights), sampleRate);
    else
      NAM_FAIL_RET(nam::Status::ErrorUnsupportedShape, "A2FastConfig: unsupported channel count", nullptr);

    // The constructor loads weights and cannot report a bad stream by returning,
    // so it latches instead. Check here, where there is somewhere to put the
    // answer, and discard a model built from a stream that did not add up --
    // otherwise it would run with zero-filled tail weights and sound wrong
    // rather than fail. With exceptions enabled the constructor threw and this
    // is never reached.
    if (!IsOk(GetLastError()))
      return nullptr;

    return out;
  }
};

// -----------------------------------------------------------------------------
// Detector helpers
// -----------------------------------------------------------------------------
bool close_to(float v, float target)
{
  return std::fabs(v - target) <= 1e-7f;
}

bool all_none_strings(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_string() || e.get<std::string>() != "none")
      return false;
  }
  return true;
}

bool all_null(const nlohmann::json& j)
{
  if (!j.is_array())
    return false;
  for (const auto& e : j)
  {
    if (!e.is_null())
      return false;
  }
  return true;
}

bool film_inactive(const nlohmann::json& layer, const char* key)
{
  auto it = layer.find(key);
  if (it == layer.end() || it->is_null())
    return true;
  if (it->is_boolean())
    return !it->get<bool>();
  if (it->is_object())
    return !it->value("active", false);
  return false;
}

} // namespace

// -----------------------------------------------------------------------------
// Public API
// -----------------------------------------------------------------------------
bool is_a2_shape(const nlohmann::json& config, int* channels)
{
  // Exactly one layer array
  auto layers_it = config.find("layers");
  if (layers_it == config.end() || !layers_it->is_array() || layers_it->size() != 1)
    return false;

  // No post-stack head
  auto head_it = config.find("head");
  if (head_it != config.end() && !head_it->is_null())
    return false;

  // No conditioning DSP. When given a non-null condition_dsp the generic WaveNet
  // builds a nested model and routes the conditioning signal through it before the
  // layer stack; the fast path has no such stage and feeds the raw input as the
  // condition. The condition DSP carries its own weights, so the parent weight
  // stream is identical with or without it and the loader cannot detect the
  // difference -- the detector must reject it here, or the fast path would silently
  // produce different audio than the model it replaces.
  auto cond_it = config.find("condition_dsp");
  if (cond_it != config.end() && !cond_it->is_null())
    return false;

  // head_scale is loaded from the trailing weight, but require the field to
  // stay schema-compatible with the generic WaveNet parser.
  auto hs_it = config.find("head_scale");
  if (hs_it == config.end() || !hs_it->is_number())
    return false;

  // in_channels defaults to 1, must be 1
  if (config.value("in_channels", 1) != 1)
    return false;

  const auto& la = (*layers_it)[0];

  if (la.value("input_size", 0) != 1)
    return false;
  if (la.value("condition_size", 0) != 1)
    return false;

  const int ch = la.value("channels", 0);
  const int bn = la.value("bottleneck", 0);
  if (ch != bn)
    return false;
  if (ch != 3 && ch != 8)
    return false;

  // kernel_sizes must match kKernelSizes exactly
  auto ks_it = la.find("kernel_sizes");
  if (ks_it == la.end() || !ks_it->is_array() || ks_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*ks_it)[i].is_number_integer() || (*ks_it)[i].get<int>() != kKernelSizes[i])
      return false;
  }

  // dilations must match kDilations exactly
  auto dl_it = la.find("dilations");
  if (dl_it == la.end() || !dl_it->is_array() || dl_it->size() != kNumLayers)
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (!(*dl_it)[i].is_number_integer() || (*dl_it)[i].get<int>() != kDilations[i])
      return false;
  }

  // activation: all LeakyReLU(0.01)
  auto act_it = la.find("activation");
  if (act_it == la.end() || !act_it->is_array() || act_it->size() != kNumLayers)
    return false;
  for (const auto& a : *act_it)
  {
    if (!a.is_object() || a.value("type", std::string()) != "LeakyReLU")
      return false;
    if (!close_to(a.value("negative_slope", 0.0f), kLeakySlope))
      return false;
  }

  // gating_mode: all "none" (or field absent)
  auto gm_it = la.find("gating_mode");
  if (gm_it != la.end() && !gm_it->is_null())
  {
    if (!all_none_strings(*gm_it) || gm_it->size() != kNumLayers)
      return false;
  }

  // Legacy boolean `gated` (the pre-gating_mode schema): the generic parser maps
  // gated==true to GATED layers, which the fast path does not implement. A genuinely
  // gated model has a larger weight stream and the loader would throw, but reject it
  // here so the boundary is enforced by the detector rather than a downstream error.
  auto gated_it = la.find("gated");
  if (gated_it != la.end() && gated_it->is_boolean() && gated_it->get<bool>())
    return false;

  // secondary_activation: all null (or field absent)
  auto sa_it = la.find("secondary_activation");
  if (sa_it != la.end() && !sa_it->is_null())
  {
    if (!all_null(*sa_it) || sa_it->size() != kNumLayers)
      return false;
  }

  // head1x1 inactive
  auto h1x1_it = la.find("head1x1");
  if (h1x1_it != la.end() && h1x1_it->is_object() && h1x1_it->value("active", false))
    return false;

  // layer1x1 active with groups=1
  auto l1x1_it = la.find("layer1x1");
  if (l1x1_it == la.end() || !l1x1_it->is_object())
    return false;
  if (!l1x1_it->value("active", false))
    return false;
  if (l1x1_it->value("groups", 1) != 1)
    return false;

  // Layer-array head rechannel: k=16, out_channels=1, bias=true
  auto lah_it = la.find("head");
  if (lah_it == la.end() || !lah_it->is_object())
    return false;
  if (lah_it->value("out_channels", 0) != 1)
    return false;
  if (lah_it->value("kernel_size", 0) != kHeadKernelSize)
    return false;
  if (lah_it->value("head_dilation", 1) != 1)
    return false;
  if (!lah_it->value("bias", false))
    return false;

  // No FiLM anywhere
  for (const char* key : {"conv_pre_film", "conv_post_film", "input_mixin_pre_film", "input_mixin_post_film",
                          "activation_pre_film", "activation_post_film", "layer1x1_post_film", "head1x1_post_film"})
  {
    if (!film_inactive(la, key))
      return false;
  }

  // No grouped convolutions
  if (la.value("groups_input", 1) != 1)
    return false;
  if (la.value("groups_input_mixin", 1) != 1)
    return false;

  // Not slimmable
  auto slim_it = la.find("slimmable");
  if (slim_it != la.end() && !slim_it->is_null())
    return false;

  if (channels)
    *channels = ch;
  return true;
}

bool is_a2_shape(const WaveNetConfig& config, int* channels)
{
  // Exactly one layer array
  if (config.layer_array_params.size() != 1)
    return false;

  // No post-stack head
  if (config.with_head || config.head_params.has_value())
    return false;

  // No conditioning DSP. Same reasoning as the JSON overload, and it matters more
  // here: the condition DSP carries its own weights, so the parent weight stream is
  // byte-identical with or without it. A binary loader cannot detect the difference
  // from the weight blob, and the fast path has no conditioning stage -- it feeds
  // the raw input as the condition. Missing this check does not fail loudly; it
  // silently produces different audio than the model it replaces.
  if (config.condition_dsp != nullptr)
    return false;

  // head_scale is not checked. The JSON overload requires the field only to keep
  // the document schema-compatible with the generic parser; the value itself is
  // always overwritten from the trailing weight, by the generic path and the fast
  // path alike, so there is nothing here that could differ.

  if (config.in_channels != 1)
    return false;

  const LayerArrayParams& la = config.layer_array_params[0];

  if (la.input_size != 1)
    return false;
  if (la.condition_size != 1)
    return false;

  const int ch = la.channels;
  if (ch != la.bottleneck)
    return false;
  if (ch != 3 && ch != 8)
    return false;

  // Per-layer kernel sizes and dilations must match the A2 pattern exactly.
  if (la.kernel_sizes.size() != static_cast<size_t>(kNumLayers))
    return false;
  if (la.dilations.size() != static_cast<size_t>(kNumLayers))
    return false;
  for (int i = 0; i < kNumLayers; i++)
  {
    if (la.kernel_sizes[i] != kKernelSizes[i])
      return false;
    if (la.dilations[i] != kDilations[i])
      return false;
  }

  // Every layer LeakyReLU(0.01)
  if (la.activation_configs.size() != static_cast<size_t>(kNumLayers))
    return false;
  for (const auto& a : la.activation_configs)
  {
    if (a.type != activations::ActivationType::LeakyReLU)
      return false;
    if (!a.negative_slope.has_value() || !close_to(*a.negative_slope, kLeakySlope))
      return false;
  }

  // No gating anywhere. This subsumes the JSON overload's separate checks on the
  // legacy `gated` boolean and on secondary_activation: the parser folds `gated`
  // into gating_modes, and Layer only ever reads secondary_activation_config when
  // the mode is GATED or BLENDED (detail.h), so under all-NONE gating a secondary
  // activation cannot affect the model's output.
  if (la.gating_modes.size() != static_cast<size_t>(kNumLayers))
    return false;
  for (const auto& gm : la.gating_modes)
  {
    if (gm != GatingMode::NONE)
      return false;
  }

  // head1x1 inactive, layer1x1 active with groups=1
  if (la.head1x1_params.active)
    return false;
  if (!la.layer1x1_params.active || la.layer1x1_params.groups != 1)
    return false;

  // Layer-array head rechannel: out_channels=1, k=16, dilation=1, bias=true
  if (la.head_size != 1)
    return false;
  if (la.head_kernel_size != kHeadKernelSize)
    return false;
  if (la.head_dilation != 1)
    return false;
  if (!la.head_bias)
    return false;

  // No FiLM anywhere
  if (la.conv_pre_film_params.active || la.conv_post_film_params.active || la.input_mixin_pre_film_params.active
      || la.input_mixin_post_film_params.active || la.activation_pre_film_params.active
      || la.activation_post_film_params.active || la._layer1x1_post_film_params.active
      || la.head1x1_post_film_params.active)
  {
    return false;
  }

  // No grouped convolutions
  if (la.groups_input != 1)
    return false;
  if (la.groups_input_mixin != 1)
    return false;

  // The JSON overload also rejects a "slimmable" field. There is no typed
  // equivalent to check: a slimmable model never becomes a WaveNetConfig in the
  // first place -- parse_config_json routes it to SlimmableWavenet, a different
  // config type entirely, so it cannot reach this function.

  if (channels)
    *channels = ch;
  return true;
}

std::unique_ptr<ModelConfig> create_a2_fast_config(const nlohmann::json& config, double sampleRate)
{
  (void)sampleRate;
  int ch = 0;
  if (!is_a2_shape(config, &ch))
    NAM_FAIL_RET(nam::Status::ErrorUnsupportedShape, "create_a2_fast_config: config does not match A2 shape", nullptr);
  return create_a2_fast_config(ch);
}

std::unique_ptr<ModelConfig> create_a2_fast_config(int channels)
{
  if (channels != 3 && channels != 8)
  {
    NAM_FAIL_RET(nam::Status::ErrorUnsupportedShape, "create_a2_fast_config: unsupported channel count", nullptr);
  }
  auto out = std::make_unique<A2FastConfig>();
  out->channels = channels;
  return out;
}

void SetKernelForNextModel(Kernel k)
{
  g_pending_kernel = static_cast<int>(k);
}

Kernel GetPendingKernel()
{
  return static_cast<Kernel>(g_pending_kernel);
}

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
