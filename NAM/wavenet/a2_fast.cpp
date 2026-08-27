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

    // Conv1D input history ring buffer, column-major (Channels rows).
    std::vector<float> history;
    std::array<float, Channels> cached_prewarm_state{};
  #if NAM_A2_RING_MODE == 1
    // pow2 ring + tail mirror. Storage = (pow2_size + max_buffer_size) cols.
    // write_pos is kept in [0, pow2_size), reads use (pos & pow2_mask) and are
    // always contiguous because cols [pow2_size, pow2_size + max_buffer_size)
    // mirror cols [0, max_buffer_size).
    int pow2_size = 0;
    int pow2_mask = 0;
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
  int _head_pow2_size = 0;
  int _head_pow2_mask = 0;
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
// Smallest power of 2 >= v (v > 0).
int next_pow2(int v)
{
  int p = 1;
  while (p < v)
    p <<= 1;
  return p;
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
    L.pow2_size = next_pow2(L.max_lookback + maxBufferSize);
    L.pow2_mask = L.pow2_size - 1;
    L.history.assign(static_cast<size_t>(Channels) * (L.pow2_size + maxBufferSize), 0.0f);
    L.write_pos = L.max_lookback;
  #else
    L.history_cols = 2 * L.max_lookback + maxBufferSize;
    L.history.assign(static_cast<size_t>(Channels) * L.history_cols, 0.0f);
    L.write_pos = L.max_lookback;
  #endif
  }

  const int head_lookback = kHeadKernelSize - 1;
  #if NAM_A2_RING_MODE == 1
  _head_pow2_size = next_pow2(head_lookback + maxBufferSize);
  _head_pow2_mask = _head_pow2_size - 1;
  _head_history.assign(static_cast<size_t>(Channels) * (_head_pow2_size + maxBufferSize), 0.0f);
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
    for (size_t column = 0; column < columns; column++)
    {
      std::copy(L.cached_prewarm_state.begin(), L.cached_prewarm_state.end(),
                L.history.begin() + static_cast<std::ptrdiff_t>(column * Channels));
    }
    L.write_pos = L.max_lookback;
  }

  const size_t head_columns = _head_history.size() / Channels;
  for (size_t column = 0; column < head_columns; column++)
  {
    std::copy(_cached_head_prewarm_state.begin(), _cached_head_prewarm_state.end(),
              _head_history.begin() + static_cast<std::ptrdiff_t>(column * Channels));
  }
  _head_write_pos = kHeadKernelSize - 1;
}

template <int Channels>
void A2FastModel<Channels>::CacheStateAsPrewarmed()
{
  for (auto& L : _layers)
  {
  #if NAM_A2_RING_MODE == 1
    const int last_column = (L.write_pos - 1) & L.pow2_mask;
  #else
    const int last_column = L.write_pos - 1;
  #endif
    std::copy_n(L.history.begin() + static_cast<std::ptrdiff_t>(last_column * Channels), Channels,
                L.cached_prewarm_state.begin());
  }

  #if NAM_A2_RING_MODE == 1
  const int last_head_column = (_head_write_pos - 1) & _head_pow2_mask;
  #else
  const int last_head_column = _head_write_pos - 1;
  #endif
  std::copy_n(_head_history.begin() + static_cast<std::ptrdiff_t>(last_head_column * Channels), Channels,
              _cached_head_prewarm_state.begin());
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
  // variable-length calls but only fires once per (pow2_size / nf) blocks.
  if (wp + nf <= L.pow2_size)
  {
    copy_block<Channels>(hist + static_cast<size_t>(wp) * Channels, src, nf);
  }
  else
  {
    const int first = L.pow2_size - wp;
    std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(nf - first) * Channels * sizeof(float));
  }
  copy_block<Channels>(hist + static_cast<size_t>(L.pow2_size) * Channels, hist, mbs);
  L.write_pos = (wp + nf) & L.pow2_mask;
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
  const float* const src = _head_sum.data();
  const int wp = _head_write_pos;
  // Same no-wrap split as _ring_write, for the same reason.
  if (wp + nf <= _head_pow2_size)
  {
    copy_block<Channels>(hist + static_cast<size_t>(wp) * Channels, src, nf);
  }
  else
  {
    const int first = _head_pow2_size - wp;
    std::memcpy(hist + static_cast<size_t>(wp) * Channels, src, static_cast<size_t>(first) * Channels * sizeof(float));
    std::memcpy(hist, src + static_cast<size_t>(first) * Channels,
                static_cast<size_t>(nf - first) * Channels * sizeof(float));
  }
  copy_block<Channels>(hist + static_cast<size_t>(_head_pow2_size) * Channels, hist, mbs);
  _head_write_pos = (wp + nf) & _head_pow2_mask;
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
  const int mask = L.pow2_mask;
  auto tap_base_phys = [&](int taps_back) { return (L.write_pos - nf - taps_back * D) & mask; };
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

// Runtime dispatcher: selects the K-specialized kernel for this layer.
// For the A2 shape the detector only admits K in {6, 15}; any other value
// here means something passed the detector that shouldn't have.
template <int Channels>
void A2FastModel<Channels>::_layer_forward(int layer_idx, const float* cond, int num_frames)
{
  Layer& L = _layers[layer_idx];
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
template <int Channels>
void A2FastModel<Channels>::_head_forward(float* output, int num_frames)
{
  _head_ring_write(num_frames);
  #if NAM_A2_FIXED_BLOCK > 0
  constexpr int nf = kFixedBlock;
  (void)num_frames; // checked against kFixedBlock in process()
  #else
  const int nf = num_frames;
  #endif
  #if NAM_A2_RING_MODE == 1
  const int mask = _head_pow2_mask;
  auto col_of = [&](int f, int k) { return (_head_write_pos - nf + f - (kHeadKernelSize - 1 - k)) & mask; };
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
  for (int f = 0; f < nf; f++)
  {
    const float x = static_cast<float>(in0[f]);
    cond[f] = x;
    float* lin = &_layer_in[static_cast<size_t>(f) * Channels];
    for (int c = 0; c < Channels; c++)
      lin[c] = _rechannel_w[c] * x;
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

} // namespace a2_fast
} // namespace wavenet
} // namespace nam

#endif // NAM_ENABLE_A2_FAST
