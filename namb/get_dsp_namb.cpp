// Binary .namb loader for NAM models
// Uses the unified create_dsp() path shared with the JSON loader

#include <algorithm>
#include <cstring>
#include <fstream>
#include <stdexcept>

#include "get_dsp_namb.h"

// NAMB_WAVENET_ONLY trims this loader to the WaveNet architecture.
//
// An architecture is pulled into the link by its registration below, not by
// anything referencing it, so a target that only loads WaveNet models still
// pays for Linear, LSTM and ConvNet. Defining this drops them; a .namb carrying
// one of those IDs then fails cleanly in the registry rather than mis-loading.
// Undefined by default, so desktop and test builds keep every architecture.

#include <NAM/activations.h>
#include <NAM/dsp.h>
#include <NAM/model_config.h>
#include <NAM/get_dsp.h>
#include <NAM/status.h>
#include <NAM/wavenet/model.h>
#include <NAM/wavenet/params.h>
#if !defined(NAMB_WAVENET_ONLY)
  #include <NAM/convnet.h>
  #include <NAM/linear.h>
  #include <NAM/lstm.h>
#endif
#if defined(NAM_ENABLE_A2_FAST)
  #include <NAM/wavenet/a2_fast.h>
#endif
#include "binary_parser_registry.h"
#include "namb_format.h"

using namespace nam::namb;

namespace
{

// =============================================================================
// Activation config reading
// =============================================================================

nam::activations::ActivationConfig read_activation_config(BinaryReader& r)
{
  nam::activations::ActivationConfig config;
  config.type = static_cast<nam::activations::ActivationType>(r.read_u8());
  uint8_t param_count = r.read_u8();

  switch (config.type)
  {
    case nam::activations::ActivationType::LeakyReLU:
      if (param_count >= 1)
      {
        config.negative_slope = r.read_f32();
        for (uint8_t i = 1; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      break;

    case nam::activations::ActivationType::PReLU:
      if (param_count == 1)
      {
        config.negative_slope = r.read_f32();
      }
      else if (param_count > 1)
      {
        std::vector<float> slopes;
        slopes.reserve(param_count);
        for (uint8_t i = 0; i < param_count; i++)
          slopes.push_back(r.read_f32());
        config.negative_slopes = std::move(slopes);
      }
      break;

    case nam::activations::ActivationType::LeakyHardtanh:
      if (param_count >= 4)
      {
        config.min_val = r.read_f32();
        config.max_val = r.read_f32();
        config.min_slope = r.read_f32();
        config.max_slope = r.read_f32();
        for (uint8_t i = 4; i < param_count; i++)
          r.read_f32(); // skip extra
      }
      else
      {
        for (uint8_t i = 0; i < param_count; i++)
          r.read_f32(); // skip
      }
      break;

    default:
      // Simple activation - skip any params
      for (uint8_t i = 0; i < param_count; i++)
        r.read_f32();
      break;
  }

  return config;
}

// =============================================================================
// FiLM params reading (4 bytes)
// =============================================================================

nam::wavenet::_FiLMParams read_film_params(BinaryReader& r)
{
  uint8_t flags = r.read_u8();
  r.read_u8(); // reserved
  uint16_t groups = r.read_u16();

  bool active = (flags & 0x01) != 0;
  bool shift = (flags & 0x02) != 0;

  return nam::wavenet::_FiLMParams(active, shift, groups);
}

// =============================================================================
// Metadata parsing
// =============================================================================

struct ParsedMetadata
{
  uint8_t version_major = 0;
  uint8_t version_minor = 0;
  uint8_t version_patch = 0;
  uint8_t meta_flags = 0;
  double sample_rate = -1.0;
  double loudness = 0.0;
  double input_level = 0.0;
  double output_level = 0.0;
};

ParsedMetadata read_metadata_block(BinaryReader& r)
{
  ParsedMetadata m;
  m.version_major = r.read_u8();
  m.version_minor = r.read_u8();
  m.version_patch = r.read_u8();
  m.meta_flags = r.read_u8();
  m.sample_rate = r.read_f64();
  m.loudness = r.read_f64();
  m.input_level = r.read_f64();
  m.output_level = r.read_f64();
  r.skip(12); // reserved
  return m;
}

// Version check on the integers the header already carries.
// nam::verify_config_version() takes a string, so using it means composing one
// only for it to be parsed straight back, and dragging in the support-checker
// path - a registry behind a std::mutex, shared_ptr, stringstream - to compare
// three small integers. Bounds come from get_dsp.h so the two checks cannot
// describe different ranges.
bool is_model_version_supported(int major, int minor, int patch)
{
  // Below the earliest supported version?
  if (major != nam::EARLIEST_SUPPORTED_NAM_FILE_VERSION_MAJOR)
  {
    if (major < nam::EARLIEST_SUPPORTED_NAM_FILE_VERSION_MAJOR)
      return false;
  }
  else if (minor != nam::EARLIEST_SUPPORTED_NAM_FILE_VERSION_MINOR)
  {
    if (minor < nam::EARLIEST_SUPPORTED_NAM_FILE_VERSION_MINOR)
      return false;
  }
  else if (patch < nam::EARLIEST_SUPPORTED_NAM_FILE_VERSION_PATCH)
  {
    return false;
  }

  // Above the latest fully supported major/minor? The core checker treats a
  // newer patch as PARTIAL rather than unsupported, so only major and minor
  // gate acceptance here -- matching CoreVersionSupportChecker::support().
  if (major > nam::LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION_MAJOR)
    return false;
  if (minor > nam::LATEST_FULLY_SUPPORTED_NAM_FILE_VERSION_MINOR)
    return false;

  return true;
}

nam::ModelMetadata to_model_metadata(const ParsedMetadata& pm)
{
  nam::ModelMetadata meta;
  meta.version = std::to_string(pm.version_major) + "." + std::to_string(pm.version_minor) + "."
                 + std::to_string(pm.version_patch);
  meta.sample_rate = pm.sample_rate;
  if (pm.meta_flags & META_HAS_LOUDNESS)
    meta.loudness = pm.loudness;
  if (pm.meta_flags & META_HAS_INPUT_LEVEL)
    meta.input_level = pm.input_level;
  if (pm.meta_flags & META_HAS_OUTPUT_LEVEL)
    meta.output_level = pm.output_level;
  return meta;
}

// =============================================================================
// Binary parsing into typed configs
// =============================================================================

// Forward declaration
std::unique_ptr<nam::ModelConfig> load_model(BinaryReader& r, const float*& weights, size_t& weight_count,
                                             const nam::ModelMetadata& meta, uint16_t format_version, nam::Status& status);

#if !defined(NAMB_WAVENET_ONLY)

// --- Linear ---

std::unique_ptr<nam::ModelConfig> load_linear(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                              const nam::ModelMetadata&, uint16_t /*format_version*/, nam::Status&)
{
  auto cfg = std::make_unique<nam::linear::LinearConfig>();
  cfg->receptive_field = r.read_i32();
  cfg->bias = r.read_u8() != 0;
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();
  r.read_u8(); // reserved
  return cfg;
}

// --- LSTM ---

std::unique_ptr<nam::ModelConfig> load_lstm(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                            const nam::ModelMetadata&, uint16_t /*format_version*/, nam::Status&)
{
  auto cfg = std::make_unique<nam::lstm::LSTMConfig>();
  cfg->num_layers = r.read_u16();
  cfg->input_size = r.read_u16();
  cfg->hidden_size = r.read_u16();
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();
  r.skip(2); // reserved
  return cfg;
}

// --- ConvNet ---

std::unique_ptr<nam::ModelConfig> load_convnet(BinaryReader& r, const float*& /*weights*/, size_t& /*weight_count*/,
                                               const nam::ModelMetadata&, uint16_t /*format_version*/, nam::Status&)
{
  auto cfg = std::make_unique<nam::convnet::ConvNetConfig>();
  cfg->channels = r.read_u16();
  cfg->batchnorm = r.read_u8() != 0;
  uint8_t num_dilations = r.read_u8();
  cfg->groups = r.read_u16();
  cfg->in_channels = r.read_u8();
  cfg->out_channels = r.read_u8();

  cfg->activation = read_activation_config(r);

  cfg->dilations.reserve(num_dilations);
  for (int i = 0; i < num_dilations; i++)
    cfg->dilations.push_back(r.read_i32());

  return cfg;
}
#endif // !NAMB_WAVENET_ONLY


// --- WaveNet ---

std::unique_ptr<nam::ModelConfig> load_wavenet(BinaryReader& r, const float*& weights, size_t& weight_count,
                                               const nam::ModelMetadata& meta, uint16_t format_version, nam::Status& status)
{
  auto wc = std::make_unique<nam::wavenet::WaveNetConfig>();
  wc->in_channels = r.read_u8();
  uint8_t has_head = r.read_u8();
  uint8_t num_layer_arrays = r.read_u8();
  uint8_t has_condition_dsp = r.read_u8();

  wc->with_head = (has_head != 0);

  // Condition DSP
  if (has_condition_dsp)
  {
    uint32_t cdsp_weight_count = r.read_u32();

    // Read condition DSP metadata (48 bytes)
    ParsedMetadata cdsp_pm = read_metadata_block(r);
    nam::ModelMetadata cdsp_meta = to_model_metadata(cdsp_pm);

    // Load condition DSP model recursively via create_dsp
    // Use local copies so load_model doesn't advance the outer pointers
    const float* cdsp_weights = weights;
    size_t cdsp_wc = cdsp_weight_count;
    auto cdsp_config = load_model(r, cdsp_weights, cdsp_wc, cdsp_meta, format_version, status);
    if (cdsp_config == nullptr)
      return nullptr;
    std::vector<float> cdsp_weight_vec(weights, weights + cdsp_weight_count);
    wc->condition_dsp = nam::create_dsp(std::move(cdsp_config), std::move(cdsp_weight_vec), cdsp_meta);

    // Advance past condition DSP weights
    weights += cdsp_weight_count;
    weight_count -= cdsp_weight_count;
  }

  // Parse layer array params
  for (int la = 0; la < num_layer_arrays; la++)
  {
    uint16_t input_size = r.read_u16();
    uint16_t condition_size = r.read_u16();
    uint16_t head_size = r.read_u16();
    uint16_t la_channels = r.read_u16();
    uint16_t bottleneck = r.read_u16();
    // v1 stores one kernel_size shared by every layer in this slot and has no
    // head kernel size or head dilation (both implicitly 1). v2 repurposes the
    // slot for head_kernel_size and carries per-layer sizes after the dilations.
    uint16_t legacy_kernel_size = 0;
    int head_kernel_size = 1;
    if (format_version >= 2)
      head_kernel_size = static_cast<int>(r.read_u16());
    else
      legacy_kernel_size = r.read_u16();

    bool head_bias = r.read_u8() != 0;
    uint8_t num_dilations = r.read_u8();
    uint16_t groups_input = r.read_u16();
    uint16_t groups_input_mixin = r.read_u16();

    int head_dilation = 1;
    if (format_version >= 2)
      head_dilation = r.read_i32();

    // layer1x1 (4 bytes)
    bool layer1x1_active = r.read_u8() != 0;
    uint16_t layer1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // head1x1 (6 bytes)
    bool head1x1_active = r.read_u8() != 0;
    uint16_t head1x1_out_channels = r.read_u16();
    uint16_t head1x1_groups = r.read_u16();
    r.read_u8(); // reserved

    // 8 FiLM params (32 bytes)
    nam::wavenet::_FiLMParams conv_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams conv_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams input_mixin_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_pre_film = read_film_params(r);
    nam::wavenet::_FiLMParams activation_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams layer1x1_post_film = read_film_params(r);
    nam::wavenet::_FiLMParams head1x1_post_film = read_film_params(r);

    // Dilations [N * int32]
    std::vector<int> dilations;
    dilations.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      dilations.push_back(r.read_i32());
    // Per-layer kernel sizes [N * int32] (v2). v1 broadcasts its single value,
    // which reproduces v1 semantics exactly.
    std::vector<int> kernel_sizes;
    kernel_sizes.reserve(num_dilations);
    if (format_version >= 2)
    {
      for (int i = 0; i < num_dilations; i++)
        kernel_sizes.push_back(r.read_i32());
    }
    else
    {
      kernel_sizes.assign(num_dilations, static_cast<int>(legacy_kernel_size));
    }

    // Activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> activation_configs;
    activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      activation_configs.push_back(read_activation_config(r));

    // Gating modes [N * uint8]
    std::vector<nam::wavenet::GatingMode> gating_modes;
    gating_modes.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
    {
      uint8_t gm = r.read_u8();
      switch (gm)
      {
        case GATING_GATED: gating_modes.push_back(nam::wavenet::GatingMode::GATED); break;
        case GATING_BLENDED: gating_modes.push_back(nam::wavenet::GatingMode::BLENDED); break;
        default: gating_modes.push_back(nam::wavenet::GatingMode::NONE); break;
      }
    }

    // Secondary activation configs [N * variable]
    std::vector<nam::activations::ActivationConfig> secondary_activation_configs;
    secondary_activation_configs.reserve(num_dilations);
    for (int i = 0; i < num_dilations; i++)
      secondary_activation_configs.push_back(read_activation_config(r));

    nam::wavenet::Layer1x1Params layer1x1_params(layer1x1_active, layer1x1_groups);
    nam::wavenet::Head1x1Params head1x1_params(head1x1_active, head1x1_out_channels, head1x1_groups);

    wc->layer_array_params.emplace_back(input_size, condition_size, head_size, head_dilation, head_kernel_size,
                                       la_channels, bottleneck, std::move(kernel_sizes), std::move(dilations),
                                       std::move(activation_configs), std::move(gating_modes),
                                       head_bias, groups_input, groups_input_mixin, layer1x1_params, head1x1_params,
                                       std::move(secondary_activation_configs), conv_pre_film, conv_post_film,
                                       input_mixin_pre_film, input_mixin_post_film, activation_pre_film,
                                       activation_post_film, layer1x1_post_film, head1x1_post_film);
  }

  // head_scale is the last weight value, but set_weights_ will overwrite it.
  // Pass 0.0f; set_weights_ will set the correct value from weights.
  wc->head_scale = 0.0f;

#if defined(NAM_ENABLE_A2_FAST)
  // create_dsp() does no shape check, so run the same detector the JSON entry
  // point runs and .namb reaches the fast path on exactly the models .nam does.
  //
  // Must run on the fully-populated wc, after condition_dsp is attached: the
  // detector has to reject a non-null condition_dsp, and checking earlier would
  // see a null pointer and wrongly admit the model.
  int a2_channels = 0;
  if (nam::wavenet::a2_fast::is_a2_shape(*wc, &a2_channels))
  {
    return nam::wavenet::a2_fast::create_a2_fast_config(a2_channels);
  }
#endif

  return wc;
}

// =============================================================================
// Static registration of binary parsers
// =============================================================================

#if !defined(NAMB_WAVENET_ONLY)
static nam::namb::BinaryConfigParserHelper _register_linear(ARCH_LINEAR, load_linear);
static nam::namb::BinaryConfigParserHelper _register_lstm(ARCH_LSTM, load_lstm);
static nam::namb::BinaryConfigParserHelper _register_convnet(ARCH_CONVNET, load_convnet);
#endif
static nam::namb::BinaryConfigParserHelper _register_wavenet(ARCH_WAVENET, load_wavenet);

// =============================================================================
// Dispatch to architecture-specific loader via registry
// =============================================================================

std::unique_ptr<nam::ModelConfig> load_model(BinaryReader& r, const float*& weights, size_t& weight_count,
                                             const nam::ModelMetadata& meta, uint16_t format_version, nam::Status& status)
{
  uint8_t arch = r.read_u8();
  r.read_u8(); // reserved
  r.read_u16(); // config_size

  return BinaryConfigParserRegistry::instance().parse(arch, r, weights, weight_count, meta, format_version, status);
}

} // anonymous namespace

// =============================================================================
// Public API
// =============================================================================

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const uint8_t* data, size_t size, nam::Status& status)
{
  status = Status::Ok;

  // Deep construction failures latch rather than return; clear before the load
  // so anything found afterwards belongs to this model and not a previous one.
  ClearLastError();

  if (data == nullptr || size < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
  {
    status = Status::ErrorTooSmall;
    return nullptr;
  }

  BinaryReader header_reader(data, FILE_HEADER_SIZE);

  // Validate magic
  const uint32_t magic = header_reader.read_u32();
  if (magic != MAGIC)
  {
    status = Status::ErrorBadMagic;
    return nullptr;
  }

  // Validate container format version
  const uint16_t version = header_reader.read_u16();
  if (version < MIN_FORMAT_VERSION || version > FORMAT_VERSION)
  {
    status = Status::ErrorUnsupportedVersion;
    return nullptr;
  }

  header_reader.read_u16(); // flags
  const uint32_t total_file_size = header_reader.read_u32();
  const uint32_t weights_offset = header_reader.read_u32();
  const uint32_t total_weight_count = header_reader.read_u32();
  header_reader.read_u32(); // model_block_size
  const uint32_t stored_checksum = header_reader.read_u32();

  if (header_reader.failed())
  {
    status = Status::ErrorTruncated;
    return nullptr;
  }

  // Validate file size
  if (size < total_file_size)
  {
    status = Status::ErrorTruncated;
    return nullptr;
  }

  // Validate CRC32
  if (compute_file_crc32(data, total_file_size) != stored_checksum)
  {
    status = Status::ErrorChecksum;
    return nullptr;
  }

  // Validate weights section. Checked before it is used as a pointer, and
  // ordered so weights_offset itself cannot be past the end either.
  const size_t expected_weights_end = static_cast<size_t>(weights_offset) + total_weight_count * sizeof(float);
  if (weights_offset < MODEL_BLOCK_OFFSET || expected_weights_end > total_file_size)
  {
    status = Status::ErrorWeightsOutOfRange;
    return nullptr;
  }

  // Read metadata block (at offset 32)
  BinaryReader meta_reader(data + FILE_HEADER_SIZE, METADATA_BLOCK_SIZE);
  const ParsedMetadata pm = read_metadata_block(meta_reader);
  const ModelMetadata meta = to_model_metadata(pm);

  // Verify model config version straight from the three integers the header
  // carries. See is_model_version_supported() for why this does not go through
  // nam::verify_config_version().
  if (!is_model_version_supported(pm.version_major, pm.version_minor, pm.version_patch))
  {
    status = Status::ErrorUnsupportedModelVersion;
    return nullptr;
  }

  // Get weight data pointer
  const float* weights = reinterpret_cast<const float*>(data + weights_offset);
  size_t weight_count = total_weight_count;

  // Read model block (at offset 80)
  const size_t model_data_size = weights_offset - MODEL_BLOCK_OFFSET;
  BinaryReader model_reader(data + MODEL_BLOCK_OFFSET, model_data_size);

  // Load model config, then construct via the unified path
  auto config = load_model(model_reader, weights, weight_count, meta, version, status);
  if (config == nullptr)
  {
    if (IsOk(status))
      status = Status::ErrorInvalidConfig;
    return nullptr;
  }
  if (model_reader.failed())
  {
    status = Status::ErrorTruncated;
    return nullptr;
  }

  std::vector<float> weight_vec(weights, weights + weight_count);
  auto dsp = create_dsp(std::move(config), std::move(weight_vec), meta, status);
  if (dsp == nullptr && IsOk(status))
    status = Status::Error;
  return dsp;
}

#if !defined(NAM_NO_EXCEPTIONS)

// Throwing wrappers. The Status overload above is the implementation; these
// only translate a failed status into an exception, so the two can never
// disagree about what counts as a failure.

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const uint8_t* data, size_t size)
{
  Status status = Status::Ok;
  auto dsp = get_dsp_namb(data, size, status);
  if (dsp == nullptr)
    throw std::runtime_error(std::string("NAMB: ") + ToString(status));
  return dsp;
}

std::unique_ptr<nam::DSP> nam::get_dsp_namb(const std::filesystem::path& filename)
{
  if (!std::filesystem::exists(filename))
    throw std::runtime_error("NAMB file doesn't exist: " + filename.string());

  // Read entire file into memory
  std::ifstream file(filename, std::ios::binary | std::ios::ate);
  if (!file.is_open())
    throw std::runtime_error("Cannot open NAMB file: " + filename.string());

  size_t file_size = file.tellg();
  file.seekg(0, std::ios::beg);

  std::vector<uint8_t> data(file_size);
  file.read(reinterpret_cast<char*>(data.data()), file_size);
  file.close();

  return get_dsp_namb(data.data(), data.size());
}

#endif // !NAM_NO_EXCEPTIONS
