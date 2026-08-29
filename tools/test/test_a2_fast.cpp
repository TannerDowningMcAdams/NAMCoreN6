// Numerical verification for the A2 fast-path WaveNet:
// for a config that matches the A2 shape, the fast path must produce the same
// output as the generic WaveNet on the same input and weights.
//
// Built only when NAM_ENABLE_A2_FAST is defined at compile time.

#if defined(NAM_ENABLE_A2_FAST)

  #include <algorithm>
  #include <cassert>
  #include <cmath>
  #include <cstdint>
  #include <iostream>
  #include <memory>
  #include <random>
  #include <string>
  #include <utility>
  #include <typeinfo>
  #include <vector>

  #include "json.hpp"

  #include "NAM/dsp.h"
  #include "NAM/wavenet/a2_fast.h"
  #include "NAM/wavenet/model.h"

  #include "allocation_tracking.h"

namespace test_a2_fast
{
namespace
{

// Build a JSON config with the A2 shape, parameterized by channel count
// (3 = A2-Lite, 8 = A2-Full). Follows the real .nam schema so both the
// strict detector and the generic parser accept it.
nlohmann::json build_a2_config(int channels)
{
  using nlohmann::json;

  json activation = json::array();
  json gating_mode = json::array();
  json secondary = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    activation.push_back({{"type", "LeakyReLU"}, {"negative_slope", nam::wavenet::a2_fast::kLeakySlope}});
    gating_mode.push_back("none");
    secondary.push_back(nullptr);
  }

  json kernel_sizes = json::array();
  json dilations = json::array();
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    kernel_sizes.push_back(nam::wavenet::a2_fast::kKernelSizes[i]);
    dilations.push_back(nam::wavenet::a2_fast::kDilations[i]);
  }

  json film_inactive = {{"active", false}, {"shift", true}, {"groups", 1}};

  json layer;
  layer["input_size"] = 1;
  layer["condition_size"] = 1;
  layer["channels"] = channels;
  layer["bottleneck"] = channels;
  layer["kernel_sizes"] = kernel_sizes;
  layer["dilations"] = dilations;
  layer["activation"] = activation;
  layer["gating_mode"] = gating_mode;
  layer["secondary_activation"] = secondary;
  layer["head"] = {{"out_channels", 1}, {"kernel_size", nam::wavenet::a2_fast::kHeadKernelSize}, {"bias", true}};
  layer["head1x1"] = {{"active", false}, {"out_channels", 1}, {"groups", 1}};
  layer["layer1x1"] = {{"active", true}, {"groups", 1}};
  layer["conv_pre_film"] = film_inactive;
  layer["conv_post_film"] = film_inactive;
  layer["input_mixin_pre_film"] = film_inactive;
  layer["input_mixin_post_film"] = film_inactive;
  layer["activation_pre_film"] = film_inactive;
  layer["activation_post_film"] = film_inactive;
  layer["layer1x1_post_film"] = film_inactive;
  layer["head1x1_post_film"] = film_inactive;
  layer["groups_input"] = 1;
  layer["groups_input_mixin"] = 1;

  json config;
  config["layers"] = json::array({layer});
  config["head_scale"] = 0.01f;
  return config;
}

// Weight count for the A2 layer array with the given channel count.
// Must match the order and sizes expected by both the generic WaveNet parser
// and A2FastModel::_load_weights.
int a2_weight_count(int channels)
{
  const int bn = channels;
  int total = /*rechannel*/ channels;
  for (int i = 0; i < nam::wavenet::a2_fast::kNumLayers; i++)
  {
    const int K = nam::wavenet::a2_fast::kKernelSizes[i];
    total += bn * channels * K + bn; // conv1d weights + bias
    total += bn; // input mixin (no bias)
    total += channels * bn + channels; // layer1x1 + bias
  }
  total += channels * nam::wavenet::a2_fast::kHeadKernelSize + 1; // head rechannel + bias
  total += 1; // trailing head_scale (read by WaveNet::set_weights_)
  return total;
}

std::vector<float> make_deterministic_weights(int count, uint32_t seed)
{
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-0.3f, 0.3f);
  std::vector<float> w(count);
  for (auto& x : w)
    x = dist(rng);
  return w;
}

std::vector<NAM_SAMPLE> make_test_input(int num_frames, double sample_rate)
{
  std::vector<NAM_SAMPLE> in(num_frames);
  // Two-tone signal so the network sees varied content.
  for (int i = 0; i < num_frames; i++)
  {
    const double t = static_cast<double>(i) / sample_rate;
    in[i] = static_cast<NAM_SAMPLE>(0.25 * std::sin(2.0 * M_PI * 220.0 * t) + 0.10 * std::sin(2.0 * M_PI * 1230.0 * t));
  }
  return in;
}

std::vector<NAM_SAMPLE> run_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  // Reset also prewarms.
  dsp.Reset(48000.0, block_size);
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  return out;
}

std::vector<NAM_SAMPLE> process_dsp(nam::DSP& dsp, const std::vector<NAM_SAMPLE>& input, int block_size)
{
  std::vector<NAM_SAMPLE> out(input.size(), static_cast<NAM_SAMPLE>(0));
  int pos = 0;
  const int total = static_cast<int>(input.size());
  while (pos < total)
  {
    const int n = std::min(block_size, total - pos);
    const NAM_SAMPLE* in_ptr = input.data() + pos;
    NAM_SAMPLE* out_ptr = out.data() + pos;
    const NAM_SAMPLE* in_arr[] = {in_ptr};
    NAM_SAMPLE* out_arr[] = {out_ptr};
    dsp.process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, n);
    pos += n;
  }
  return out;
}

void compare(const std::vector<NAM_SAMPLE>& a, const std::vector<NAM_SAMPLE>& b, int channels, int block_size,
             double tol)
{
  assert(a.size() == b.size());
  double max_diff = 0.0;
  int max_i = 0;
  for (size_t i = 0; i < a.size(); i++)
  {
    const double d = std::fabs(static_cast<double>(a[i]) - static_cast<double>(b[i]));
    if (d > max_diff)
    {
      max_diff = d;
      max_i = static_cast<int>(i);
    }
  }
  if (!(max_diff < tol))
  {
    std::cerr << "A2FastModel<" << channels << "> diverges from generic WaveNet "
              << "(block=" << block_size << "): max |diff| = " << max_diff << " at i=" << max_i
              << " (generic=" << a[max_i] << ", fast=" << b[max_i] << ")" << std::endl;
    assert(false);
  }
}

} // namespace

void test_detector_matches_lite()
{
  auto cfg = build_a2_config(3);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 3);
}

void test_detector_matches_full()
{
  auto cfg = build_a2_config(8);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

void test_detector_accepts_nonstandard_head_scale()
{
  auto cfg = build_a2_config(8);
  cfg["head_scale"] = 0.0042f;
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

void test_detector_rejects_wrong_channels()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["channels"] = 4;
  cfg["layers"][0]["bottleneck"] = 4;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_wrong_kernel_sizes()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["kernel_sizes"][0] = 7; // tweak first entry
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_wrong_activation()
{
  auto cfg = build_a2_config(8);
  cfg["layers"][0]["activation"][0] = {{"type", "Tanh"}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_detector_rejects_gating()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["gating_mode"][0] = "gated";
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

// A condition DSP routes the conditioning signal through a nested model; the fast
// path has no such stage. The nested model holds its own weights, so the parent
// weight stream is unchanged and only the detector can catch this -- otherwise the
// fast path would silently produce different audio than the generic WaveNet.
void test_detector_rejects_condition_dsp()
{
  auto cfg = build_a2_config(8);
  cfg["condition_dsp"] = {{"version", "0.5.0"},
                          {"architecture", "Linear"},
                          {"config", nlohmann::json::object()},
                          {"weights", nlohmann::json::array()}};
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

// Legacy boolean `gated` (pre-gating_mode schema) maps to GATED layers in the
// generic parser, which the fast path does not implement.
void test_detector_rejects_legacy_gated()
{
  auto cfg = build_a2_config(3);
  cfg["layers"][0]["gated"] = true;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_matches_generic(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  // Fast path: build through the explicit A2 factory.
  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Generic path: call parse_config_json directly so the dispatcher's A2 shortcut
  // doesn't kick in. This gives us the reference WaveNet against the same config.
  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  // Exercise with two block sizes to catch any off-by-one in the ring-buffer rewind.
  const int total = 2048;
  const auto input = make_test_input(total, 48000.0);
  for (int block : {64, 256})
  {
    const auto out_generic = run_dsp(*generic_dsp, input, block);
    const auto out_fast = run_dsp(*fast_dsp, input, block);
    // 1e-5 is what nam2c_verify.py uses as its byte-exactness threshold; allow a little
    // slack because the generic path sums through Eigen and may reorder FMAs.
    compare(out_generic, out_fast, channels, block, /*tol=*/5e-5);
  }
}

void test_matches_generic_lite()
{
  test_matches_generic(3);
}

void test_matches_generic_full()
{
  test_matches_generic(8);
}

// The fast path must report the same prewarm count as the generic WaveNet it
// replaces; otherwise Reset() warms the two by a different number of samples and
// their first post-Reset output can diverge (regression guard for the A2 prewarm
// off-by-one). Builds both from the identical config via the same dual path as
// test_matches_generic.
void test_prewarm_matches_generic(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_gen = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_gen), 48000.0);

  assert(fast_dsp->GetPrewarmSamples() == generic_dsp->GetPrewarmSamples());
}

void test_prewarm_matches_generic_lite()
{
  test_prewarm_matches_generic(3);
}

void test_prewarm_matches_generic_full()
{
  test_prewarm_matches_generic(8);
}

// With no cache, Reset() uses the legacy silence-processing prewarm and caches
// its steady state. Process more than a receptive field of audio to disturb every
// convolution history, restore the cached prewarm, then require the same audio to
// produce exactly the same output as it did after the legacy prewarm.
void test_cached_prewarm_dsp(nam::DSP& dsp, int channels, const std::string& implementation)
{
  const int block_size = 64;
  dsp.Reset(48000.0, block_size);
  const auto input = make_test_input(dsp.GetPrewarmSamples() + block_size, 48000.0);
  const auto expected = process_dsp(dsp, input, block_size);

  // Cached restoration must not fall back to DSP::prewarm(), which allocates
  // silence buffers and processes the full receptive field.
  const std::string test_name = implementation + "<" + std::to_string(channels) + ">::cached prewarm";
  allocation_tracking::run_allocation_test_no_allocations(
    nullptr, [&]() { dsp.prewarm(); }, nullptr, test_name.c_str());

  const auto actual = process_dsp(dsp, input, block_size);
  compare(expected, actual, channels, block_size, /*tol=*/1.0e-12);
}

void test_cached_prewarm(int channels)
{
  const auto cfg = build_a2_config(channels);
  const auto weights = make_deterministic_weights(a2_weight_count(channels), /*seed=*/0xA2CA000u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);
  test_cached_prewarm_dsp(*fast_dsp, channels, "A2FastModel");

  auto generic_cfg = nam::wavenet::parse_config_json(cfg, 48000.0);
  std::vector<float> w_generic = weights;
  auto generic_dsp = generic_cfg.create(std::move(w_generic), 48000.0);
  test_cached_prewarm_dsp(*generic_dsp, channels, "WaveNet");
}

void test_cached_prewarm_lite()
{
  test_cached_prewarm(3);
}

void test_cached_prewarm_full()
{
  test_cached_prewarm(8);
}

// Real-time safety: once the DSP has been Reset (buffers sized, prewarmed),
// subsequent process() calls must not allocate or free heap memory. Uses the
// same allocation-tracking infrastructure as the generic WaveNet RT-safety
// tests (tools/test/allocation_tracking.{h,cpp}) — overridden malloc/free and
// global new/delete increment counters while tracking is enabled.
void test_process_realtime_safe(int channels)
{
  const auto cfg = build_a2_config(channels);
  const int weight_count = a2_weight_count(channels);
  const auto weights = make_deterministic_weights(weight_count, /*seed=*/0xA2FA500u + channels);

  auto fast_cfg = nam::wavenet::a2_fast::create_a2_fast_config(cfg, 48000.0);
  std::vector<float> w_fast = weights;
  auto fast_dsp = fast_cfg->create(std::move(w_fast), 48000.0);

  // Exercise several block sizes all within a single pre-sized state so the
  // internal "num_frames > max_buffer_size" guard in process() never fires
  // (which would legitimately reallocate).
  const int max_buffer = 256;
  fast_dsp->Reset(48000.0, max_buffer);

  const int total = 4 * max_buffer;
  const auto input = make_test_input(total, 48000.0);
  std::vector<NAM_SAMPLE> output(total, 0.0);

  // Warm up caches / any lazy init with one untracked pass.
  {
    const NAM_SAMPLE* in = input.data();
    NAM_SAMPLE* out = output.data();
    const NAM_SAMPLE* in_arr[] = {in};
    NAM_SAMPLE* out_arr[] = {out};
    fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, max_buffer);
  }

  for (int block : {1, 32, 64, 128, 256})
  {
    std::string test_name = "A2FastModel<" + std::to_string(channels) + ">::process block=" + std::to_string(block);
    allocation_tracking::run_allocation_test_no_allocations(
      nullptr,
      [&]() {
        int pos = 0;
        while (pos + block <= total)
        {
          const NAM_SAMPLE* in = input.data() + pos;
          NAM_SAMPLE* out = output.data() + pos;
          const NAM_SAMPLE* in_arr[] = {in};
          NAM_SAMPLE* out_arr[] = {out};
          fast_dsp->process(const_cast<NAM_SAMPLE**>(in_arr), out_arr, block);
          pos += block;
        }
      },
      nullptr, test_name.c_str());
  }
}

void test_process_realtime_safe_lite()
{
  test_process_realtime_safe(3);
}

void test_process_realtime_safe_full()
{
  test_process_realtime_safe(8);
}


// =============================================================================
// Typed detector: is_a2_shape(const WaveNetConfig&, int*)
//
// create_config() consults the JSON overload but create_dsp() does not, so a
// loader that builds a WaveNetConfig directly - the .namb loader - reaches the
// generic WaveNet whatever shape the model is. These cover the typed overload
// that closes that gap.
//
// The parity test matters most: it pins the two detectors to the same verdict,
// so a predicate added to one and not the other fails here rather than silently
// changing which class a .namb model gets.
// =============================================================================

namespace
{

// Parse the shared JSON fixture into the typed config a binary loader would
// otherwise have built by hand.
nam::wavenet::WaveNetConfig parse_a2_config(int channels)
{
  return nam::wavenet::parse_config_json(build_a2_config(channels), 48000.0);
}

// Minimal stand-in for a conditioning model. nam::DSP has no pure virtuals, so
// this needs no behaviour -- the detector only tests the pointer for null, and
// building a real nested model here would couple the test to another
// architecture's weight layout.
class StubConditionDSP : public nam::DSP
{
public:
  StubConditionDSP()
  : nam::DSP(/*in_channels=*/1, /*out_channels=*/1, /*expected_sample_rate=*/48000.0)
  {
  }
};

// Both detectors must agree on every config, accepted or rejected.
void assert_detectors_agree(const nlohmann::json& cfg, const char* what)
{
  int json_ch = -1;
  int typed_ch = -2;
  const bool json_match = nam::wavenet::a2_fast::is_a2_shape(cfg, &json_ch);

  auto typed = nam::wavenet::parse_config_json(cfg, 48000.0);
  const bool typed_match = nam::wavenet::a2_fast::is_a2_shape(typed, &typed_ch);

  if (json_match != typed_match)
  {
    std::cerr << "Detector disagreement on " << what << ": json=" << json_match << " typed=" << typed_match
              << std::endl;
    assert(false);
  }
  if (json_match)
  {
    assert(json_ch == typed_ch);
  }
}

} // namespace

void test_typed_detector_matches_lite()
{
  auto cfg = parse_a2_config(3);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 3);
}

void test_typed_detector_matches_full()
{
  auto cfg = parse_a2_config(8);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, &ch));
  assert(ch == 8);
}

// The check that motivated the typed overload. A condition DSP carries its own
// weights, so the parent weight stream is byte-identical with or without one --
// nothing downstream of the detector can catch the substitution, and the fast
// path has no conditioning stage. A binary loader attaches condition_dsp after
// parsing the layer arrays, so this has to hold on the fully populated config,
// not on a partially built one.
void test_typed_detector_rejects_condition_dsp()
{
  auto cfg = parse_a2_config(8);
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr)); // accepted before
  cfg.condition_dsp = std::make_unique<StubConditionDSP>();
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr)); // rejected after
}

void test_typed_detector_rejects_post_stack_head()
{
  auto cfg = parse_a2_config(3);
  assert(nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
  cfg.with_head = true;
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_typed_detector_rejects_wrong_channels()
{
  auto j = build_a2_config(3);
  j["layers"][0]["channels"] = 4;
  j["layers"][0]["bottleneck"] = 4;
  auto cfg = nam::wavenet::parse_config_json(j, 48000.0);
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_typed_detector_rejects_wrong_kernel_sizes()
{
  auto j = build_a2_config(8);
  j["layers"][0]["kernel_sizes"][0] = 7;
  auto cfg = nam::wavenet::parse_config_json(j, 48000.0);
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

void test_typed_detector_rejects_gating()
{
  auto j = build_a2_config(3);
  j["layers"][0]["gating_mode"][0] = "gated";
  j["layers"][0]["secondary_activation"][0] = {{"type", "Sigmoid"}};
  auto cfg = nam::wavenet::parse_config_json(j, 48000.0);
  assert(!nam::wavenet::a2_fast::is_a2_shape(cfg, nullptr));
}

// Parity across the whole fixture set. Any predicate present in one detector and
// missing from the other shows up here.
void test_typed_detector_agrees_with_json()
{
  assert_detectors_agree(build_a2_config(3), "A2-Lite");
  assert_detectors_agree(build_a2_config(8), "A2-Full");

  {
    auto j = build_a2_config(8);
    j["head_scale"] = 0.0042f;
    assert_detectors_agree(j, "nonstandard head_scale");
  }
  {
    auto j = build_a2_config(3);
    j["layers"][0]["channels"] = 4;
    j["layers"][0]["bottleneck"] = 4;
    assert_detectors_agree(j, "wrong channels");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["kernel_sizes"][0] = 7;
    assert_detectors_agree(j, "wrong kernel_sizes");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["dilations"][5] = 99;
    assert_detectors_agree(j, "wrong dilations");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["activation"][0] = {{"type", "Tanh"}};
    assert_detectors_agree(j, "wrong activation");
  }
  {
    auto j = build_a2_config(3);
    j["layers"][0]["gating_mode"][0] = "gated";
    j["layers"][0]["secondary_activation"][0] = {{"type", "Sigmoid"}};
    assert_detectors_agree(j, "gating");
  }
  // NOTE: legacy `gated` is deliberately absent from this list. See
  // test_typed_detector_diverges_on_contradictory_gating below.
  {
    auto j = build_a2_config(8);
    j["layers"][0]["layer1x1"] = {{"active", false}, {"groups", 1}};
    assert_detectors_agree(j, "layer1x1 inactive");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["head1x1"] = {{"active", true}, {"out_channels", 8}, {"groups", 1}};
    assert_detectors_agree(j, "head1x1 active");
  }
  {
    auto j = build_a2_config(3);
    j["layers"][0]["conv_post_film"] = {{"active", true}, {"shift", true}, {"groups", 1}};
    assert_detectors_agree(j, "FiLM active");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["groups_input"] = 2;
    assert_detectors_agree(j, "grouped input conv");
  }
  {
    auto j = build_a2_config(8);
    j["layers"][0]["head"] = {{"out_channels", 1}, {"kernel_size", 8}, {"bias", true}};
    assert_detectors_agree(j, "wrong head kernel_size");
  }
}

// The one config the two detectors legitimately disagree on, pinned here so the
// divergence is a documented property rather than a surprise.
//
// A layer carrying both gating_mode (all "none") and the legacy gated=true is
// self-contradictory. parse_config_json resolves it by precedence - gating_mode
// present means the boolean is never read - so the parsed model has no gating
// and the fast path substitutes for it validly.
//
//   JSON detector  rejects  - tests gated unconditionally, without the
//                             precedence rule. A missed optimisation only.
//   Typed detector accepts  - correct about what the model actually does.
//
// The typed overload could not reproduce the JSON verdict anyway: `gated` does
// not survive parsing. This is the safe direction to differ in; the reverse is
// what test_typed_detector_rejects_condition_dsp guards.
void test_typed_detector_diverges_on_contradictory_gating()
{
  auto j = build_a2_config(3);
  j["layers"][0]["gated"] = true;

  assert(!nam::wavenet::a2_fast::is_a2_shape(j, nullptr)); // JSON: conservative reject

  auto typed = nam::wavenet::parse_config_json(j, 48000.0);
  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(typed, &ch)); // typed: accept
  assert(ch == 3);

  // The accept is justified: the parser really did produce an ungated model.
  for (const auto& gm : typed.layer_array_params[0].gating_modes)
  {
    assert(gm == nam::wavenet::GatingMode::NONE);
  }
}

// The channel-count overload is what a binary loader calls once the typed
// detector has reported a match. It must refuse anything the fast path has no
// template instantiation for, rather than handing back an unusable config.
void test_create_a2_fast_config_from_channels()
{
  for (const int ch : {3, 8})
  {
    auto cfg = nam::wavenet::a2_fast::create_a2_fast_config(ch);
    assert(cfg != nullptr);
  }

  bool threw = false;
  try
  {
    (void)nam::wavenet::a2_fast::create_a2_fast_config(4);
  }
  catch (const std::runtime_error&)
  {
    threw = true;
  }
  assert(threw);
}

// End-to-end, with no JSON anywhere in the substitution path: a config reached
// the way a binary loader reaches it must instantiate A2FastModel rather than
// the generic WaveNet, and must stay numerically faithful to it. This is the
// property that was broken -- a2_probe reported GENERIC_WAVENET for .namb on a
// model that .nam ran as A2_FAST.
void test_typed_detector_selects_fast_path(int channels)
{
  auto wc = parse_a2_config(channels);

  int ch = 0;
  assert(nam::wavenet::a2_fast::is_a2_shape(wc, &ch));
  assert(ch == channels);

  auto fast_config = nam::wavenet::a2_fast::create_a2_fast_config(ch);
  const auto weights = make_deterministic_weights(a2_weight_count(channels), 0x5EED);

  auto fast = fast_config->create(weights, 48000.0);
  std::unique_ptr<nam::DSP> generic(new nam::wavenet::WaveNet(
    wc.in_channels, wc.layer_array_params, wc.head_scale, wc.with_head, std::nullopt, weights, nullptr, 48000.0));

  assert(fast != nullptr);
  assert(generic != nullptr);
  // Distinct concrete types: the substitution actually happened.
  assert(typeid(*fast) != typeid(*generic));

  const int num_frames = 2048;
  const int max_buffer = 256;
  fast->Reset(48000.0, max_buffer);
  generic->Reset(48000.0, max_buffer);

  auto input = make_test_input(num_frames, 48000.0);
  std::vector<NAM_SAMPLE> out_fast(num_frames, 0.0);
  std::vector<NAM_SAMPLE> out_generic(num_frames, 0.0);

  for (int offset = 0; offset < num_frames; offset += max_buffer)
  {
    const int n = std::min(max_buffer, num_frames - offset);
    NAM_SAMPLE* in_ptr = input.data() + offset;
    NAM_SAMPLE* f_ptr = out_fast.data() + offset;
    NAM_SAMPLE* g_ptr = out_generic.data() + offset;
    fast->process(&in_ptr, &f_ptr, n);
    generic->process(&in_ptr, &g_ptr, n);
  }

  double max_diff = 0.0;
  for (int i = 0; i < num_frames; i++)
  {
    max_diff = std::max(max_diff, std::abs(static_cast<double>(out_fast[i]) - static_cast<double>(out_generic[i])));
  }

  if (!(max_diff < 1e-5))
  {
    std::cerr << "test_typed_detector_selects_fast_path(" << channels << "): max_diff=" << max_diff << std::endl;
    assert(false);
  }
}

void test_typed_detector_selects_fast_path_lite()
{
  test_typed_detector_selects_fast_path(3);
}

void test_typed_detector_selects_fast_path_full()
{
  test_typed_detector_selects_fast_path(8);
}
} // namespace test_a2_fast

#endif // NAM_ENABLE_A2_FAST
