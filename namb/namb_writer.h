#pragma once
// .nam (JSON) -> .namb (binary), without building a DOM.
//
// Reads the source a character at a time through lwjson's streaming parser,
// fills the plain structs in namb_config.h, and emits through the byte-verified
// Emit* functions there. It replaced a DOM front end that produced the same
// bytes, and is held to the same goldens -- the ones the original host pipeline
// wrote, which every pack ever flashed to a pedal was built from.
//
//   source bytes --[lwjson]--> LayerArrayCfg &c. --[Emit*]--> .namb bytes
//
// WHY. A DOM of an A2 container is ~1 MB of nodes to extract an 8 KB .namb, and
// nlohmann tears that down through a heap vector sized by the array being
// destroyed -- 224 KB against a 216 KB heap, which is an abort() under
// -fno-exceptions. The extraction was always a single forward pass; the DOM was
// only what a convenient API handed us.
//
// Nothing here allocates. lwjson's streaming parser has a fixed stack and no
// malloc, the parse state lives in this object, and the source is never
// buffered beyond a single character.
//
// COST. sizeof(StreamConverter) is ~16 KB, dominated by the LayerArrayCfg it
// holds. That is a caller-owned object, not a stack local -- the target's MSP
// stack is 8 KiB -- so ModelLibrary keeps one as a member.
//
//   nam::namb::StreamConverter conv;   // a member on the firmware
//   nam::namb::WriteResult r;
//   FileSource src{...};
//   nam::Status st = nam::namb::WriteNambStream(src, blob, sizeof(blob), r, opts, conv);
//
// SOURCE CONTRACT. Any type with `int get()` returning the next byte as 0..255,
// or a negative value at end of input. SdFs-backed on the firmware, a file or a
// memory span on the host.
//
// WHAT IT WILL NOT DO. `condition_dsp` is refused rather than half-supported:
// its weights must appear in the output before the outer model's, but they are
// encountered in the source before the model block that precedes them is even
// sized, so a single pass cannot place them without buffering an array whose
// length is not ours to bound. No model has ever carried one, and get_dsp_namb
// is built NAMB_WAVENET_ONLY. A second pass over the source is the answer if
// that ever changes.

#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <NAM/status.h>

#include "namb_config.h"
#include "namb_format.h"

extern "C" {
#include "lwjson/lwjson.h"
}

// The deepest path this grammar reaches is an activation object's key inside an
// activation array inside a layer array inside a container's submodel:
//
//   root > config > submodels > [i] > model > config > layers > [j]
//        > activation > [k] > type
//
// Every key occupies a stack entry of its own alongside its object or array, so
// that bottoms out at 17. lwjson's default is 16, and overflowing it is not a
// loud failure -- prv_stack_push returns 0, the parse comes back lwjsonERRMEM,
// and every container on the card reports "not valid JSON". So the setting in
// Dependencies/lwjson/lwjson/lwjson_opts.h is load-bearing, and a build that
// does not pick that file up has to fail here rather than at run time.
static_assert(LWJSON_CFG_STREAM_STACK_SIZE >= 24,
              "lwjson stack is too shallow for a SlimmableContainer -- is lwjson_opts.h on the "
              "include path? See NAMCoreN6/Dependencies/lwjson/README.md.");

namespace nam
{
namespace namb
{

namespace stream_detail
{

/// \brief Sentinel for "this anchor is not currently in play".
static constexpr size_t kNoAnchor = static_cast<size_t>(-1);

/// \brief Does the token look like a JSON integer rather than a real?
///
/// The same split nlohmann's lexer makes, and the reason it matters is below in
/// TokenToF32.
inline bool TokenIsIntegral(const char* t)
{
  for (const char* p = t; *p != '\0'; p++)
  {
    if (*p == '.' || *p == 'e' || *p == 'E')
      return false;
  }
  return true;
}

/// \brief A weight token as the .namb float32 it becomes.
///
/// This is the one function in the rewrite that had to be derived rather than
/// transcribed, because the goldens pin a *double* rounding:
///
///   nlohmann's lexer converts the token with std::strtod into a double
///   (json.hpp:8311), and WeightsInOrder then narrowed that to float with
///   v.get<float>().
///
/// strtof() rounds decimal -> float in one step and disagrees with
/// decimal -> double -> float on some inputs. Using it here would corrupt a
/// handful of weights out of every fourteen thousand, which would read as an
/// unrelated bug. So: strtod, then narrow.
///
/// The integer branch is the other half of the same fidelity. nlohmann stored a
/// fractionless token as int64 and get<float> converted straight from it, never
/// through a double -- so an integer-typed weight is converted the same way
/// here.
inline float TokenToF32(const char* t)
{
  if (TokenIsIntegral(t))
  {
    errno = 0;
    const long long v = std::strtoll(t, nullptr, 10);
    if (errno != ERANGE)
      return static_cast<float>(v);
    // Out of int64 range: nlohmann falls back to a float too.
  }
  return static_cast<float>(std::strtod(t, nullptr));
}

/// Copy with truncation, which is what every caller here wants: a metadata
/// name is sanitised down to 31 characters afterwards, and a version string
/// longer than "255.255.255" is nonsense already. snprintf would do it too,
/// but GCC cannot see the bound and warns on every call.
inline void CopyBounded(char* dst, size_t dst_size, const char* src)
{
  size_t i = 0;
  for (; i + 1 < dst_size && src[i] != '\0'; i++)
    dst[i] = src[i];
  dst[i] = '\0';
}

inline double TokenToF64(const char* t)
{
  return std::strtod(t, nullptr);
}

/// \brief Integer conversion matching the DOM path's AsInt: a whole number
///        written as a real is accepted, a fractional one is not.
inline bool TokenToInt(const char* t, int& out)
{
  if (TokenIsIntegral(t))
  {
    errno = 0;
    const long long v = std::strtoll(t, nullptr, 10);
    if (errno == ERANGE)
      return false;
    out = static_cast<int>(v);
    return true;
  }

  const double d = std::strtod(t, nullptr);
  const double truncated = static_cast<double>(static_cast<long long>(d));
  if (d != truncated)
    return false;
  out = static_cast<int>(static_cast<long long>(d));
  return true;
}

/// Order matches nam::activations::ActivationType. Both LeakyHardtanh
/// spellings are accepted upstream, so both are accepted here.
inline bool ActivationTypeFromName(const char* name, uint8_t& out)
{
  struct Entry
  {
    const char* name;
    uint8_t id;
  };
  static const Entry kTable[] = {
    {"Tanh", 0},      {"Hardtanh", 1},      {"Fasttanh", 2},      {"ReLU", 3},
    {"LeakyReLU", 4}, {"PReLU", 5},         {"Sigmoid", 6},       {"SiLU", 7},
    {"Hardswish", 8}, {"LeakyHardtanh", 9}, {"LeakyHardTanh", 9}, {"Softsign", 10},
  };
  for (const Entry& e : kTable)
  {
    if (std::strcmp(e.name, name) == 0)
    {
      out = e.id;
      return true;
    }
  }
  return false;
}

inline bool GatingModeFromName(const char* s, uint8_t& out)
{
  if (std::strcmp(s, "none") == 0)
  {
    out = GATING_NONE;
    return true;
  }
  if (std::strcmp(s, "gated") == 0)
  {
    out = GATING_GATED;
    return true;
  }
  if (std::strcmp(s, "blended") == 0)
  {
    out = GATING_BLENDED;
    return true;
  }
  return false;
}

/// Parses "major.minor.patch". Missing components are zero, which is what
/// nam2namb's sscanf produced for a short version string.
inline void ParseVersion(const char* s, uint8_t out[3])
{
  out[0] = out[1] = out[2] = 0;
  for (int part = 0; part < 3 && *s != '\0'; part++)
  {
    unsigned v = 0;
    if (*s < '0' || *s > '9')
      break;
    while (*s >= '0' && *s <= '9')
    {
      if (v < 1000u) // saturate rather than wrap on a nonsense version
        v = v * 10u + static_cast<unsigned>(*s - '0');
      s++;
    }
    out[part] = static_cast<uint8_t>(v > 255u ? 255u : v);
    if (*s == '.')
      s++;
    else
      break;
  }
}

static constexpr uint8_t kActivationSigmoid = 6;
static constexpr uint8_t kActivationTanh = 0;

/// The eight FiLM blocks, in the order the format writes them.
static const char* const kFilmKeys[kNumFilmBlocks] = {
  "conv_pre_film",       "conv_post_film",       "input_mixin_pre_film", "input_mixin_post_film",
  "activation_pre_film", "activation_post_film", "layer1x1_post_film",   "head1x1_post_film",
};

} // namespace stream_detail

// =============================================================================
// StreamConverter
// =============================================================================

/// \brief Converts one .nam to one .namb, a character at a time.
///
/// Reusable: begin() resets everything, so a caller can hold one of these for
/// the life of the program and convert model after model through it.
class StreamConverter
{
public:
  StreamConverter() = default;
  StreamConverter(const StreamConverter&) = delete;
  StreamConverter& operator=(const StreamConverter&) = delete;

  /// \brief Bind an output span and start a new conversion.
  Status begin(uint8_t* out, size_t capacity, const WriteOptions& opts, WriteResult& result);

  /// \brief Feed one source byte.
  /// \return The first error the conversion hit, or Status::Ok.
  Status feed(char c);

  /// \brief True once the model of interest has been fully read. The caller can
  ///        stop reading the source -- for a channels=3 import that is usually
  ///        a seventh of the file.
  bool complete() const { return complete_; }

  /// \brief Backpatch the header, write the metadata block and derive the name.
  Status finish();

private:
  using Cfg = LayerArrayCfg;

  // ---- lwjson plumbing ----------------------------------------------------
  static void Trampoline(lwjson_stream_parser_t* jsp, lwjson_stream_type_t type)
  {
    static_cast<StreamConverter*>(lwjson_stream_get_user_data(jsp))->onEvent(type);
  }
  void onEvent(lwjson_stream_type_t type);

  // ---- stack queries ------------------------------------------------------
  //
  // At a value event lwjson has the owning KEY still on top, so the value of a
  // key sitting at stack index i fires with stack_pos == i + 1. Object and
  // array *start* events fire before the push, and end events after the pop, so
  // both see the index the container itself occupies. Every predicate below is
  // that arithmetic, named.

  size_t depth() const { return jsp_.stack_pos; }

  bool isKey(size_t i, const char* k) const
  {
    return i < jsp_.stack_pos && jsp_.stack[i].type == LWJSON_STREAM_TYPE_KEY
           && std::strcmp(jsp_.stack[i].meta.name, k) == 0;
  }

  bool isArray(size_t i) const
  {
    return i < jsp_.stack_pos && jsp_.stack[i].type == LWJSON_STREAM_TYPE_ARRAY;
  }

  const char* keyAt(size_t i) const
  {
    return (i < jsp_.stack_pos && jsp_.stack[i].type == LWJSON_STREAM_TYPE_KEY) ? jsp_.stack[i].meta.name : nullptr;
  }

  /// The current value belongs to key \p k at stack index \p i.
  bool valueOf(size_t i, const char* k) const { return jsp_.stack_pos == i + 1 && isKey(i, k); }

  /// The current value is an element of the array at stack index \p i.
  bool elementOf(size_t i) const { return jsp_.stack_pos == i + 1 && isArray(i); }

  uint16_t elementIndex(size_t i) const { return jsp_.stack[i].meta.index; }

  const char* str() const { return jsp_.data.str.buff; }
  const char* prim() const { return jsp_.data.prim.buff; }

  // ---- failure ------------------------------------------------------------
  bool fail(Status st, const char* fmt, ...);
  bool failed() const { return !IsOk(status_); }

  // ---- document structure -------------------------------------------------
  void onObjectStart();
  void onObjectEnd();
  void onArrayStart();
  void onArrayEnd();
  void onValue(lwjson_stream_type_t type);

  void enterModel(size_t model_key);
  void endModel();
  void beginLayer();
  void endLayer();
  bool finalizeLayer();
  void closeConfig();
  void noteConfigMember(bool is_null);

  bool decideOnFirstLayer();
  void startEmitting();

  // ---- value routing ------------------------------------------------------
  bool routeRootValue(lwjson_stream_type_t type);
  bool routeModelValue(lwjson_stream_type_t type);
  bool routeLayerValue(lwjson_stream_type_t type);
  bool routeActivationValue(lwjson_stream_type_t type);

  void beginActivation(bool secondary, bool is_array, uint16_t index);
  bool finishActivation();
  bool activationFromName(const char* name, ActivationCfg& out);

  // =========================================================================
  // State
  // =========================================================================

  lwjson_stream_parser_t jsp_ = {};
  Status status_ = Status::Ok;
  bool begun_ = false; ///< A '{' has been seen; BOM bytes before it are skipped
  bool complete_ = false;
  bool json_done_ = false;

  uint8_t* out_ = nullptr;
  size_t capacity_ = 0;
  WriteResult* result_ = nullptr;
  WriteOptions opts_;

  SpanWriter w_{nullptr, 0};
  FileHeaderPatch header_patch_;
  ModelBlockPatch block_patch_;
  size_t model_block_start_ = 0;
  size_t model_block_size_ = 0;
  size_t wavenet_header_off_ = 0;
  size_t weights_offset_ = 0;
  uint32_t weight_count_ = 0;
  bool emitting_ = false; ///< The chosen model is being written out
  bool block_closed_ = false;

  // ---- anchors ----
  size_t submodels_arr_ = stream_detail::kNoAnchor;
  size_t model_key_ = stream_detail::kNoAnchor;
  size_t cfg_key_ = stream_detail::kNoAnchor;
  size_t layers_arr_ = stream_detail::kNoAnchor;
  size_t layer_key_ = stream_detail::kNoAnchor;

  bool is_container_ = false;
  bool skip_model_ = false; ///< This submodel is not the one asked for
  bool decided_ = false; ///< The chosen model has been settled on
  bool model_done_ = false; ///< The chosen model's object has closed
  bool root_meta_seen_ = false; ///< The container's own metadata object closed

  // ---- captured metadata ----
  struct MetaCapture
  {
    bool has[3] = {}; ///< loudness, input_level_dbu, output_level_dbu
    double val[3] = {};
    bool name_set = false;
    char name[96] = {};
  };
  MetaCapture outer_; ///< The container's, folded in behind the submodel's
  MetaCapture inner_; ///< The converted model's own

  bool version_set_ = false;
  char version_[24] = {};
  double sample_rate_ = -1.0;

  // ---- WaveNet config header ----
  uint8_t in_channels_ = 1;
  uint8_t has_head_ = 0;
  uint8_t layer_array_count_ = 0;

  // ---- what the container had on offer, for the failure sentence ----
  char available_[48] = {};
  size_t available_len_ = 0;

  // =========================================================================
  // Per-layer parse state
  // =========================================================================
  //
  // A layer array's keys arrive in whatever order the document holds them, and
  // several fields default from others -- bottleneck from channels, head1x1's
  // out_channels from channels -- so nothing is resolved until the layer object
  // closes. These flags are what "was it actually present" looks like without a
  // DOM to ask.

  struct LayerParse
  {
    uint16_t dilations = 0;
    bool channels_set = false;
    int channels = 0;
    bool input_size_set = false;
    int input_size = 0;
    bool condition_size_set = false;
    int condition_size = 0;

    bool bottleneck_set = false;
    int bottleneck = 0;
    int groups_input = 1;
    int groups_input_mixin = 1;

    bool head_obj = false;
    bool head_out_channels_set = false;
    int head_out_channels = 0;
    bool head_kernel_set = false;
    int head_kernel = 1;
    bool head_bias_set = false;
    bool head_bias = false;
    int head_dilation = 1;
    bool legacy_head_size_set = false;
    int legacy_head_size = 0;
    bool legacy_head_bias_set = false;
    bool legacy_head_bias = false;

    bool ks_array = false;
    uint16_t ks_count = 0;
    bool ks_scalar_set = false;
    int ks_scalar = 0;

    bool layer1x1_active = true;
    int layer1x1_groups = 1;
    bool head1x1_active = false;
    bool head1x1_out_set = false;
    int head1x1_out = 0;
    int head1x1_groups = 1;

    // FiLM: resolved to flags as each block closes, defaults for absent ones.
    bool film_seen[kNumFilmBlocks] = {};
    bool film_is_bool[kNumFilmBlocks] = {};
    bool film_bool[kNumFilmBlocks] = {};
    bool film_active[kNumFilmBlocks] = {};
    bool film_shift[kNumFilmBlocks] = {};
    int film_groups[kNumFilmBlocks] = {};

    // Gating: an array of strings, one broadcast string, or the legacy bool.
    enum class Gating : uint8_t
    {
      Absent,
      Array,
      Scalar,
      Legacy
    };
    Gating gating = Gating::Absent;
    uint16_t gating_count = 0;
    uint8_t gating_scalar = GATING_NONE;

    // Activations. "single" is the broadcast form; the array form lands
    // straight in the LayerArrayCfg as each element finishes.
    bool act_seen = false;
    bool act_array = false;
    uint16_t act_count = 0;
    ActivationCfg act_single;

    bool sec_seen = false;
    bool sec_array = false;
    uint16_t sec_count = 0;
    ActivationCfg sec_single;
    /// 0 absent, 1 explicitly null, 2 parsed. A null secondary is legal as long
    /// as its dilation's gating is NONE, which is how every shipping model is
    /// written -- so it cannot be rejected on sight.
    uint8_t sec_single_kind = 0;
    uint8_t sec_kind[kMaxDilations] = {};
  };
  LayerParse lp_;

  // ---- the activation currently being read ----
  struct ActParse
  {
    bool type_set = false;
    uint8_t type = 0;
    bool leaky_set = false;
    double leaky = 0.01;
    bool prelu_scalar_set = false;
    double prelu_scalar = 0.0;
    bool slopes_array = false;
    uint16_t slopes_off = 0;
    uint16_t slopes_n = 0;
    bool lht_set[4] = {};
    double lht[4] = {-1.0, 1.0, 0.01, 0.01};
  };
  ActParse act_;

  enum class ActSlot : uint8_t
  {
    None,
    Primary,
    Secondary
  };
  ActSlot act_slot_ = ActSlot::None;
  bool act_in_array_ = false;
  uint16_t act_index_ = 0;
  size_t act_obj_idx_ = stream_detail::kNoAnchor; ///< Stack index of the activation object

  ModelScratch scratch_;
};

// =============================================================================
// Lifecycle
// =============================================================================

inline Status StreamConverter::begin(uint8_t* out, size_t capacity, const WriteOptions& opts, WriteResult& result)
{
  result = WriteResult{};

  out_ = out;
  capacity_ = capacity;
  result_ = &result;
  opts_ = opts;

  status_ = Status::Ok;
  begun_ = false;
  complete_ = false;
  json_done_ = false;

  w_ = SpanWriter(out, capacity);
  header_patch_ = FileHeaderPatch{};
  block_patch_ = ModelBlockPatch{};
  model_block_start_ = 0;
  model_block_size_ = 0;
  wavenet_header_off_ = 0;
  weights_offset_ = 0;
  weight_count_ = 0;
  emitting_ = false;
  block_closed_ = false;

  submodels_arr_ = stream_detail::kNoAnchor;
  model_key_ = stream_detail::kNoAnchor;
  cfg_key_ = stream_detail::kNoAnchor;
  layers_arr_ = stream_detail::kNoAnchor;
  layer_key_ = stream_detail::kNoAnchor;

  is_container_ = false;
  skip_model_ = false;
  decided_ = false;
  model_done_ = false;
  root_meta_seen_ = false;

  outer_ = MetaCapture{};
  inner_ = MetaCapture{};
  version_set_ = false;
  version_[0] = '\0';
  sample_rate_ = -1.0;

  in_channels_ = 1;
  has_head_ = 0;
  layer_array_count_ = 0;

  available_[0] = '\0';
  available_len_ = 0;

  act_slot_ = ActSlot::None;
  act_obj_idx_ = stream_detail::kNoAnchor;

  if (out == nullptr || capacity < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
  {
    return FailImpl(result.detail, Status::ErrorTooSmall,
                    "output span is too small to hold even a .namb header");
  }

  lwjson_stream_init(&jsp_, &StreamConverter::Trampoline);
  lwjson_stream_set_user_data(&jsp_, this);
  return Status::Ok;
}

inline Status StreamConverter::feed(char c)
{
  if (failed() || complete_ || json_done_)
    return status_;

  // A UTF-8 BOM is not whitespace to lwjson, which would refuse the document on
  // its first byte. Half the models on hand carry one.
  if (!begun_)
  {
    if (static_cast<unsigned char>(c) == 0xEF || static_cast<unsigned char>(c) == 0xBB
        || static_cast<unsigned char>(c) == 0xBF)
      return status_;
    if (c == '{' || c == '[')
      begun_ = true;
  }

  const lwjsonr_t r = lwjson_stream_parse(&jsp_, c);
  if (failed())
    return status_;

  if (r == lwjsonSTREAMDONE)
  {
    json_done_ = true;
    return status_;
  }
  if (r != lwjsonSTREAMINPROG && r != lwjsonSTREAMWAITFIRSTCHAR)
  {
    fail(Status::ErrorInvalidConfig, "not valid JSON");
  }
  return status_;
}

inline bool StreamConverter::fail(Status st, const char* fmt, ...)
{
  if (failed())
    return false; // keep the first reason, which is the one that explains it

  va_list args;
  va_start(args, fmt);
  std::vsnprintf(result_->detail, kDetailSize, fmt, args);
  va_end(args);
  status_ = st;
  return false;
}

// =============================================================================
// Event dispatch
// =============================================================================

inline void StreamConverter::onEvent(lwjson_stream_type_t type)
{
  if (failed() || complete_)
    return;

  switch (type)
  {
    case LWJSON_STREAM_TYPE_OBJECT: onObjectStart(); break;
    case LWJSON_STREAM_TYPE_OBJECT_END: onObjectEnd(); break;
    case LWJSON_STREAM_TYPE_ARRAY: onArrayStart(); break;
    case LWJSON_STREAM_TYPE_ARRAY_END: onArrayEnd(); break;
    case LWJSON_STREAM_TYPE_KEY: break; // keys are read off the stack, not here
    case LWJSON_STREAM_TYPE_STRING:
    case LWJSON_STREAM_TYPE_NUMBER:
    case LWJSON_STREAM_TYPE_TRUE:
    case LWJSON_STREAM_TYPE_FALSE:
    case LWJSON_STREAM_TYPE_NULL: onValue(type); break;
    default: break;
  }
}

inline void StreamConverter::enterModel(size_t model_key)
{
  model_key_ = model_key;
  cfg_key_ = model_key + 2;
  layers_arr_ = model_key + 3;
  layer_key_ = model_key + 5;

  inner_ = MetaCapture{};
  version_set_ = false;
  version_[0] = '\0';
  sample_rate_ = -1.0;
  in_channels_ = 1;
  has_head_ = 0;
  layer_array_count_ = 0;
  skip_model_ = false;
  block_closed_ = false;
}

inline void StreamConverter::onObjectStart()
{
  const size_t p = depth();

  // The root. Provisionally the model too: a bare WaveNet document is its own
  // model, and if "architecture" later says SlimmableContainer we undo this.
  if (p == 0)
  {
    enterModel(1);
    return;
  }

  // submodels[i].model -- the stack here is
  // root(0) config(1) OBJ(2) submodels(3) ARR(4) elem(5) model(6) OBJ(7).
  if (is_container_ && submodels_arr_ != stream_detail::kNoAnchor && !decided_
      && p == submodels_arr_ + 3 && isKey(submodels_arr_ + 2, "model"))
  {
    enterModel(p + 1);
    return;
  }

  if (model_key_ == stream_detail::kNoAnchor || skip_model_)
    return;

  // A layer array object.
  if (layers_arr_ != stream_detail::kNoAnchor && p == layers_arr_ + 1 && isArray(layers_arr_))
  {
    beginLayer();
    return;
  }

  // A config member whose mere presence is what the format records.
  if (cfg_key_ != stream_detail::kNoAnchor && p == cfg_key_ + 1)
    noteConfigMember(false);

  // Sub-objects of a layer array. Only that they were there is recorded here;
  // their members arrive as ordinary values.
  if (layer_key_ != stream_detail::kNoAnchor && p == layer_key_ + 1)
  {
    if (const char* k = keyAt(layer_key_))
    {
      if (std::strcmp(k, "head") == 0)
        lp_.head_obj = true;
      else
      {
        for (size_t i = 0; i < kNumFilmBlocks; i++)
        {
          if (std::strcmp(k, stream_detail::kFilmKeys[i]) == 0)
          {
            lp_.film_seen[i] = true;
            lp_.film_is_bool[i] = false;
            break;
          }
        }
      }
    }
  }

  // An activation object, either the broadcast form or one array element.
  if (layer_key_ != stream_detail::kNoAnchor)
  {
    if (p == layer_key_ + 1)
    {
      if (isKey(layer_key_, "activation"))
      {
        beginActivation(false, false, 0);
        act_obj_idx_ = p;
      }
      else if (isKey(layer_key_, "secondary_activation"))
      {
        beginActivation(true, false, 0);
        act_obj_idx_ = p;
      }
    }
    else if (p == layer_key_ + 2 && isArray(layer_key_ + 1))
    {
      if (isKey(layer_key_, "activation"))
      {
        beginActivation(false, true, elementIndex(layer_key_ + 1));
        act_obj_idx_ = p;
      }
      else if (isKey(layer_key_, "secondary_activation"))
      {
        beginActivation(true, true, elementIndex(layer_key_ + 1));
        act_obj_idx_ = p;
      }
    }
  }
}

inline void StreamConverter::onObjectEnd()
{
  const size_t p = depth(); // index the closed object occupied

  if (act_slot_ != ActSlot::None && p == act_obj_idx_)
  {
    finishActivation();
    return;
  }

  // The root's own metadata object closed. For a container that is the last
  // thing the conversion can still be waiting on.
  if (p == 2 && isKey(1, "metadata"))
  {
    root_meta_seen_ = true;
    if (model_done_)
      complete_ = true;
  }

  if (model_key_ == stream_detail::kNoAnchor)
    return;

  // A layer array finished.
  if (!skip_model_ && layers_arr_ != stream_detail::kNoAnchor && p == layers_arr_ + 1 && isArray(layers_arr_))
  {
    endLayer();
    return;
  }

  // The model's config finished: everything the model block needs is known.
  if (!skip_model_ && p == model_key_ + 1 && isKey(model_key_, "config"))
  {
    closeConfig();
    return;
  }

  // The model itself finished.
  if (p + 1 == model_key_)
    endModel();
}

inline void StreamConverter::onArrayStart()
{
  const size_t p = depth();

  // root.config.submodels
  if (is_container_ && p == 4 && isKey(3, "submodels") && isKey(1, "config"))
  {
    submodels_arr_ = p;
    return;
  }

  if (model_key_ == stream_detail::kNoAnchor || skip_model_)
    return;

  // model.config.layers
  if (p == cfg_key_ + 1 && isKey(cfg_key_, "layers"))
    return; // layers_arr_ is already model_key_+3 by construction

  // model.weights -- the point at which the model block must already be done.
  if (p == model_key_ + 1 && isKey(model_key_, "weights"))
  {
    if (!emitting_)
      return;
    if (!block_closed_)
    {
      fail(Status::ErrorInvalidConfig,
           "'weights' appears before 'config'; this converter reads the source in one pass");
      return;
    }
    w_.align_to(4);
    weights_offset_ = w_.position();
    return;
  }

  // A config member that happens to be an array still counts as present.
  if (cfg_key_ != stream_detail::kNoAnchor && p == cfg_key_ + 1)
    noteConfigMember(false);

  // Arrays hanging off a layer array. Marking the shape here rather than on the
  // first element is what makes an empty one -- "kernel_sizes": [] -- come out
  // as a length mismatch rather than as an absent key.
  if (layer_key_ != stream_detail::kNoAnchor && p == layer_key_ + 1)
  {
    if (const char* k = keyAt(layer_key_))
    {
      if (std::strcmp(k, "dilations") == 0)
        lp_.dilations = 0;
      else if (std::strcmp(k, "kernel_sizes") == 0)
      {
        lp_.ks_array = true;
        lp_.ks_count = 0;
      }
      else if (std::strcmp(k, "gating_mode") == 0)
      {
        lp_.gating = LayerParse::Gating::Array;
        lp_.gating_count = 0;
      }
      else if (std::strcmp(k, "activation") == 0)
      {
        lp_.act_seen = true;
        lp_.act_array = true;
        lp_.act_count = 0;
      }
      else if (std::strcmp(k, "secondary_activation") == 0)
      {
        lp_.sec_seen = true;
        lp_.sec_array = true;
        lp_.sec_count = 0;
      }
    }
  }

  // A PReLU's per-channel slopes.
  if (act_slot_ != ActSlot::None && act_obj_idx_ != stream_detail::kNoAnchor && p == act_obj_idx_ + 2
      && isKey(act_obj_idx_ + 1, "negative_slopes"))
  {
    act_.slopes_array = true;
    act_.slopes_n = 0;
  }
}

inline void StreamConverter::onArrayEnd()
{
  const size_t p = depth();

  // The submodels list ran out without the requested channel count. Checked
  // before the anchor guard below: rejecting a submodel retires those anchors,
  // so by the time the list closes there is no model in play by definition.
  if (is_container_ && !decided_ && submodels_arr_ != stream_detail::kNoAnchor && p == submodels_arr_)
  {
    fail(Status::ErrorInvalidConfig, "container has no channels=%u submodel (it has %s)",
         static_cast<unsigned>(opts_.channels), available_len_ ? available_ : "none");
    return;
  }

  if (model_key_ == stream_detail::kNoAnchor)
    return;

  if (skip_model_)
    return;

  if (layers_arr_ != stream_detail::kNoAnchor && p == layers_arr_ && isKey(cfg_key_, "layers"))
  {
    if (layer_array_count_ == 0)
      fail(Status::ErrorInvalidConfig, "WaveNet config needs a non-empty 'layers' array");
    return;
  }

  // Per-layer arrays: fix up the counts that had to be counted as they arrived.
  if (layer_key_ != stream_detail::kNoAnchor && p == layer_key_ + 1)
  {
    if (isKey(layer_key_, "dilations"))
      scratch_.layer.num_dilations = static_cast<uint8_t>(lp_.dilations);
  }
}

// =============================================================================
// Choosing a model and starting to emit
// =============================================================================

inline bool StreamConverter::decideOnFirstLayer()
{
  const int ch = lp_.channels_set ? lp_.channels : -1;

  if (is_container_)
  {
    if (ch > 0 && available_len_ + 8 < sizeof(available_))
    {
      available_len_ += static_cast<size_t>(std::snprintf(available_ + available_len_,
                                                          sizeof(available_) - available_len_,
                                                          available_len_ ? ", %d" : "%d", ch));
    }
    if (ch < 0 || static_cast<uint16_t>(ch) != opts_.channels)
    {
      skip_model_ = true;
      return false;
    }
    result_->channels = opts_.channels;
  }
  else
  {
    result_->channels = (ch > 0) ? static_cast<uint16_t>(ch) : 0;
  }

  decided_ = true;
  startEmitting();
  return true;
}

inline void StreamConverter::startEmitting()
{
  EmitFileHeader(w_, header_patch_);

  // The metadata block sits at a fixed offset and a fixed size, and one of its
  // fields -- sample_rate -- is written after the weights in every model on
  // hand. Leave the room and fill it in finish(), through a SpanWriter aimed at
  // those 48 bytes.
  w_.write_zeros(METADATA_BLOCK_SIZE);

  model_block_start_ = w_.position();
  block_patch_ = EmitModelBlockOpen(w_, ARCH_WAVENET);

  // in_channels, has_head and the layer-array count are not all known yet --
  // config keys may follow "layers" -- so the four bytes are placed now and
  // corrected when the config object closes.
  wavenet_header_off_ = w_.position();
  EmitWaveNetConfigHeader(w_, WaveNetCfg{});

  emitting_ = true;
}

inline void StreamConverter::closeConfig()
{
  if (!emitting_)
    return;

  WaveNetCfg cfg;
  cfg.in_channels = in_channels_;
  cfg.has_head = has_head_;
  cfg.num_layer_arrays = layer_array_count_;
  cfg.has_condition_dsp = 0; // a condition_dsp is refused outright, see the header

  SpanWriter hw(out_ + wavenet_header_off_, 4);
  EmitWaveNetConfigHeader(hw, cfg);

  size_t config_size = 0;
  if (!EmitModelBlockClose(w_, block_patch_, config_size))
  {
    fail(Status::ErrorInvalidConfig, "config block is %u bytes, too large for its uint16 length",
         static_cast<unsigned>(config_size));
    return;
  }

  model_block_size_ = w_.position() - model_block_start_;
  block_closed_ = true;
}

inline void StreamConverter::endModel()
{
  // Either way the anchors are retired: a container's next submodel sits at the
  // same depths as this one, and leaving them set would route its fields into a
  // model that is already finished.
  model_key_ = stream_detail::kNoAnchor;
  cfg_key_ = stream_detail::kNoAnchor;
  layers_arr_ = stream_detail::kNoAnchor;
  layer_key_ = stream_detail::kNoAnchor;
  skip_model_ = false;

  if (!emitting_)
    return; // a submodel we did not want; keep looking

  model_done_ = true;

  // Reading on is usually waste -- a channels=3 import has no use for the
  // channels=8 weights behind it -- but the container's metadata supplies
  // input_level_dbu and output_level_dbu, and Deluxe_Reverb.nam puts its
  // metadata object *after* the submodels. Stopping before that costs two of
  // the three level fields .namb can carry. So the early exit is taken only
  // once there is nothing left to learn.
  if (!is_container_ || root_meta_seen_)
    complete_ = true;
}

// =============================================================================
// Value routing
// =============================================================================

inline void StreamConverter::onValue(lwjson_stream_type_t type)
{
  // Deepest first. An activation object's keys ("type", "negative_slope") sit
  // at the same depth as a head's, and only the slot tells them apart.
  if (act_slot_ != ActSlot::None && routeActivationValue(type))
    return;

  if (routeRootValue(type))
    return;

  if (model_key_ == stream_detail::kNoAnchor || skip_model_)
    return;

  if (routeModelValue(type))
    return;

  routeLayerValue(type);
}

inline bool StreamConverter::routeRootValue(lwjson_stream_type_t type)
{
  // root.architecture decides whether the root is the model or a wrapper.
  if (valueOf(1, "architecture"))
  {
    if (type != LWJSON_STREAM_TYPE_STRING)
    {
      fail(Status::ErrorInvalidConfig, "document has no string 'architecture'");
      return true;
    }

    if (std::strcmp(str(), "SlimmableContainer") == 0)
    {
      is_container_ = true;
      // What was captured as the model's own is the container's after all; the
      // submodels supply their own. The two copies were kept in step precisely
      // so this costs nothing -- "architecture" follows "metadata" in every
      // model on hand, and JSON does not promise otherwise.
      inner_ = MetaCapture{};
      version_set_ = false;
      model_key_ = stream_detail::kNoAnchor;
      cfg_key_ = stream_detail::kNoAnchor;
      layers_arr_ = stream_detail::kNoAnchor;
      layer_key_ = stream_detail::kNoAnchor;
    }
    else if (std::strcmp(str(), "WaveNet") != 0)
    {
      fail(Status::ErrorUnknownArchitecture,
           "architecture '%s' is not supported; expected WaveNet or SlimmableContainer", str());
    }
    return true;
  }

  // root.metadata.* -- kept whether or not this turns out to be a container,
  // because we do not know yet.
  if (depth() == 4 && isKey(1, "metadata"))
  {
    const char* k = keyAt(3);
    if (k == nullptr)
      return false;

    if (type == LWJSON_STREAM_TYPE_NUMBER)
    {
      static const char* const kKeys[3] = {"loudness", "input_level_dbu", "output_level_dbu"};
      for (int i = 0; i < 3; i++)
      {
        if (std::strcmp(k, kKeys[i]) == 0)
        {
          outer_.has[i] = true;
          outer_.val[i] = stream_detail::TokenToF64(prim());
        }
      }
    }
    else if (type == LWJSON_STREAM_TYPE_STRING && std::strcmp(k, "name") == 0)
    {
      stream_detail::CopyBounded(outer_.name, sizeof(outer_.name), str());
      outer_.name_set = true;
    }
  }

  return false; // a bare model's root keys are also its model keys
}

inline void StreamConverter::noteConfigMember(bool is_null)
{
  if (is_null)
    return; // a null member is an absent one, as it was through the DOM

  if (isKey(cfg_key_, "head"))
    has_head_ = 1;
  else if (isKey(cfg_key_, "condition_dsp"))
  {
    fail(Status::ErrorInvalidConfig,
         "this model has a condition_dsp, which the streaming converter cannot place in one pass");
  }
}

inline bool StreamConverter::routeModelValue(lwjson_stream_type_t type)
{
  const size_t m = model_key_;

  if (valueOf(m, "version"))
  {
    if (type == LWJSON_STREAM_TYPE_STRING)
    {
      stream_detail::CopyBounded(version_, sizeof(version_), str());
      version_set_ = true;
    }
    return true;
  }

  if (valueOf(m, "sample_rate"))
  {
    if (type != LWJSON_STREAM_TYPE_NUMBER)
    {
      fail(Status::ErrorInvalidConfig, "sample_rate must be a number");
      return true;
    }
    sample_rate_ = stream_detail::TokenToF64(prim());
    return true;
  }

  // A submodel's own architecture. The root's was handled above.
  if (m != 1 && valueOf(m, "architecture"))
  {
    if (type != LWJSON_STREAM_TYPE_STRING)
      fail(Status::ErrorInvalidConfig, "model has no string 'architecture'");
    else if (std::strcmp(str(), "WaveNet") != 0)
      fail(Status::ErrorUnknownArchitecture,
           "architecture '%s' is not supported; this build converts WaveNet only", str());
    return true;
  }

  // model.metadata.*
  if (depth() == m + 3 && isKey(m, "metadata"))
  {
    const char* k = keyAt(m + 2);
    if (k == nullptr)
      return true;

    if (type == LWJSON_STREAM_TYPE_NUMBER)
    {
      static const char* const kKeys[3] = {"loudness", "input_level_dbu", "output_level_dbu"};
      for (int i = 0; i < 3; i++)
      {
        if (std::strcmp(k, kKeys[i]) == 0)
        {
          inner_.has[i] = true;
          inner_.val[i] = stream_detail::TokenToF64(prim());
        }
      }
    }
    else if (type == LWJSON_STREAM_TYPE_STRING && std::strcmp(k, "name") == 0)
    {
      stream_detail::CopyBounded(inner_.name, sizeof(inner_.name), str());
      inner_.name_set = true;
    }
    return true;
  }

  // model.config.*
  if (depth() == cfg_key_ + 1)
  {
    if (isKey(cfg_key_, "in_channels"))
    {
      int v = 0;
      if (type != LWJSON_STREAM_TYPE_NUMBER || !stream_detail::TokenToInt(prim(), v))
        fail(Status::ErrorInvalidConfig, "config.in_channels must be an integer");
      else
        in_channels_ = static_cast<uint8_t>(v);
      return true;
    }
    if (isKey(cfg_key_, "head") || isKey(cfg_key_, "condition_dsp"))
    {
      noteConfigMember(type == LWJSON_STREAM_TYPE_NULL);
      return true;
    }
  }

  // model.weights[]
  if (isKey(m, "weights") && elementOf(m + 1))
  {
    if (!emitting_)
      return true;
    if (type != LWJSON_STREAM_TYPE_NUMBER)
    {
      fail(Status::ErrorInvalidConfig, "'weights' must be an array of numbers");
      return true;
    }
    w_.write_f32(stream_detail::TokenToF32(prim()));
    weight_count_++;
    return true;
  }

  return false;
}

inline bool StreamConverter::activationFromName(const char* name, ActivationCfg& out)
{
  out = ActivationCfg{};
  if (!stream_detail::ActivationTypeFromName(name, out.type))
    return fail(Status::ErrorInvalidConfig, "layer array %u: unknown activation '%s'",
                static_cast<unsigned>(layer_array_count_), name);
  return true; // the string form carries no parameters
}

inline bool StreamConverter::routeLayerValue(lwjson_stream_type_t type)
{
  const size_t l = layer_key_;
  if (l == stream_detail::kNoAnchor)
    return false;

  const unsigned la_n = static_cast<unsigned>(layer_array_count_);
  const bool is_num = (type == LWJSON_STREAM_TYPE_NUMBER);
  const bool is_true = (type == LWJSON_STREAM_TYPE_TRUE);
  const bool is_false = (type == LWJSON_STREAM_TYPE_FALSE);
  const bool is_str = (type == LWJSON_STREAM_TYPE_STRING);
  const bool is_null = (type == LWJSON_STREAM_TYPE_NULL);

  // ---- scalars sitting directly on the layer object ----
  if (depth() == l + 1)
  {
    const char* k = keyAt(l);
    if (k == nullptr)
      return false;

    if (is_num)
    {
      int v = 0;
      if (!stream_detail::TokenToInt(prim(), v))
        return fail(Status::ErrorInvalidConfig, "layer array %u: '%s' is not an integer", la_n, k);

      if (std::strcmp(k, "input_size") == 0) { lp_.input_size = v; lp_.input_size_set = true; }
      else if (std::strcmp(k, "condition_size") == 0) { lp_.condition_size = v; lp_.condition_size_set = true; }
      else if (std::strcmp(k, "channels") == 0) { lp_.channels = v; lp_.channels_set = true; }
      else if (std::strcmp(k, "bottleneck") == 0) { lp_.bottleneck = v; lp_.bottleneck_set = true; }
      else if (std::strcmp(k, "groups_input") == 0) { lp_.groups_input = v; }
      else if (std::strcmp(k, "groups_input_mixin") == 0) { lp_.groups_input_mixin = v; }
      else if (std::strcmp(k, "kernel_size") == 0) { lp_.ks_scalar = v; lp_.ks_scalar_set = true; }
      else if (std::strcmp(k, "head_size") == 0) { lp_.legacy_head_size = v; lp_.legacy_head_size_set = true; }
      return true;
    }

    if (is_true || is_false)
    {
      if (std::strcmp(k, "head_bias") == 0)
      {
        lp_.legacy_head_bias = is_true;
        lp_.legacy_head_bias_set = true;
        return true;
      }
      if (std::strcmp(k, "gated") == 0)
      {
        if (lp_.gating == LayerParse::Gating::Absent)
        {
          lp_.gating = LayerParse::Gating::Legacy;
          lp_.gating_scalar = is_true ? GATING_GATED : GATING_NONE;
        }
        return true;
      }
      // "conv_pre_film": false is shorthand for absent. The true spelling has
      // never appeared, but reading it as "active with defaults" is the only
      // sensible meaning if it ever does.
      for (size_t i = 0; i < kNumFilmBlocks; i++)
      {
        if (std::strcmp(k, stream_detail::kFilmKeys[i]) == 0)
        {
          lp_.film_seen[i] = true;
          lp_.film_is_bool[i] = true;
          lp_.film_bool[i] = is_true;
          return true;
        }
      }
      return true;
    }

    if (is_str)
    {
      if (std::strcmp(k, "gating_mode") == 0)
      {
        uint8_t mode = GATING_NONE;
        if (!stream_detail::GatingModeFromName(str(), mode))
          return fail(Status::ErrorInvalidConfig, "layer array %u: bad gating_mode", la_n);
        lp_.gating = LayerParse::Gating::Scalar;
        lp_.gating_scalar = mode;
        return true;
      }
      if (std::strcmp(k, "activation") == 0)
      {
        lp_.act_seen = true;
        lp_.act_array = false;
        return activationFromName(str(), lp_.act_single);
      }
      if (std::strcmp(k, "secondary_activation") == 0)
      {
        lp_.sec_seen = true;
        lp_.sec_array = false;
        lp_.sec_single_kind = 2;
        return activationFromName(str(), lp_.sec_single);
      }
      return true;
    }

    if (is_null)
    {
      // Absent, throughout -- including a null secondary_activation, which only
      // matters for dilations whose gating is not NONE.
      if (std::strcmp(k, "secondary_activation") == 0)
      {
        lp_.sec_seen = true;
        lp_.sec_array = false;
        lp_.sec_single_kind = 1;
      }
      return true;
    }
    return true;
  }

  // ---- elements of an array sitting directly on the layer object ----
  if (depth() == l + 2 && isArray(l + 1))
  {
    const char* k = keyAt(l);
    if (k == nullptr)
      return false;
    const uint16_t i = elementIndex(l + 1);

    if (std::strcmp(k, "dilations") == 0)
    {
      if (i >= kMaxDilations)
        return fail(Status::ErrorInvalidConfig, "layer array %u: more than %u dilations, the format's limit", la_n,
                    static_cast<unsigned>(kMaxDilations));
      int v = 0;
      if (!is_num || !stream_detail::TokenToInt(prim(), v))
        return fail(Status::ErrorInvalidConfig, "layer array %u: dilations must all be integers", la_n);
      scratch_.layer.dilations[i] = static_cast<int32_t>(v);
      lp_.dilations = static_cast<uint16_t>(i + 1);
      return true;
    }

    if (std::strcmp(k, "kernel_sizes") == 0)
    {
      if (i >= kMaxDilations)
        return fail(Status::ErrorInvalidConfig, "layer array %u: too many kernel_sizes", la_n);
      int v = 0;
      if (!is_num || !stream_detail::TokenToInt(prim(), v))
        return fail(Status::ErrorInvalidConfig, "layer array %u: kernel_sizes[%u] is not an integer", la_n,
                    static_cast<unsigned>(i));
      scratch_.layer.kernel_sizes[i] = static_cast<int32_t>(v);
      lp_.ks_count = static_cast<uint16_t>(i + 1);
      return true;
    }

    if (std::strcmp(k, "gating_mode") == 0)
    {
      if (i >= kMaxDilations)
        return fail(Status::ErrorInvalidConfig, "layer array %u: too many gating modes", la_n);
      uint8_t mode = GATING_NONE;
      if (!is_str || !stream_detail::GatingModeFromName(str(), mode))
        return fail(Status::ErrorInvalidConfig, "layer array %u: bad gating_mode at index %u", la_n,
                    static_cast<unsigned>(i));
      scratch_.layer.gating[i] = mode;
      lp_.gating_count = static_cast<uint16_t>(i + 1);
      return true;
    }

    if (std::strcmp(k, "activation") == 0)
    {
      if (i >= kMaxDilations)
        return fail(Status::ErrorInvalidConfig, "layer array %u: too many activations", la_n);
      if (is_str)
      {
        if (!activationFromName(str(), scratch_.layer.activations[i]))
          return false;
        lp_.act_count = static_cast<uint16_t>(i + 1);
        return true;
      }
      return fail(Status::ErrorInvalidConfig, "layer array %u activation %u: activation must be a string or an object",
                  la_n, static_cast<unsigned>(i));
    }

    if (std::strcmp(k, "secondary_activation") == 0)
    {
      if (i >= kMaxDilations)
        return fail(Status::ErrorInvalidConfig, "layer array %u: too many secondary activations", la_n);
      lp_.sec_count = static_cast<uint16_t>(i + 1);
      if (is_null)
      {
        lp_.sec_kind[i] = 1; // legal as long as this dilation's gating is NONE
        return true;
      }
      if (is_str)
      {
        if (!activationFromName(str(), scratch_.layer.secondary[i]))
          return false;
        lp_.sec_kind[i] = 2;
        return true;
      }
      return true; // an object element; finishActivation marks it
    }
    return true;
  }

  // ---- members of head / layer1x1 / head1x1 / a FiLM block ----
  if (depth() == l + 3 && !isArray(l + 1))
  {
    const char* parent = keyAt(l);
    const char* k = keyAt(l + 2);
    if (parent == nullptr || k == nullptr)
      return false;

    int v = 0;
    const bool as_int = is_num && stream_detail::TokenToInt(prim(), v);

    if (std::strcmp(parent, "head") == 0)
    {
      if (std::strcmp(k, "out_channels") == 0 && as_int) { lp_.head_out_channels = v; lp_.head_out_channels_set = true; }
      else if (std::strcmp(k, "kernel_size") == 0 && as_int) { lp_.head_kernel = v; lp_.head_kernel_set = true; }
      else if (std::strcmp(k, "head_dilation") == 0 && as_int) { lp_.head_dilation = v; }
      else if (std::strcmp(k, "bias") == 0 && (is_true || is_false)) { lp_.head_bias = is_true; lp_.head_bias_set = true; }
      return true;
    }

    if (std::strcmp(parent, "layer1x1") == 0)
    {
      if (std::strcmp(k, "active") == 0 && (is_true || is_false)) lp_.layer1x1_active = is_true;
      else if (std::strcmp(k, "groups") == 0 && as_int) lp_.layer1x1_groups = v;
      else if (is_num && !as_int)
        return fail(Status::ErrorInvalidConfig, "layer array %u: layer1x1 is malformed", la_n);
      return true;
    }

    if (std::strcmp(parent, "head1x1") == 0)
    {
      if (std::strcmp(k, "active") == 0 && (is_true || is_false)) lp_.head1x1_active = is_true;
      else if (std::strcmp(k, "out_channels") == 0 && as_int) { lp_.head1x1_out = v; lp_.head1x1_out_set = true; }
      else if (std::strcmp(k, "groups") == 0 && as_int) lp_.head1x1_groups = v;
      else if (is_num && !as_int)
        return fail(Status::ErrorInvalidConfig, "layer array %u: head1x1 is malformed", la_n);
      return true;
    }

    for (size_t i = 0; i < kNumFilmBlocks; i++)
    {
      if (std::strcmp(parent, stream_detail::kFilmKeys[i]) != 0)
        continue;
      if (std::strcmp(k, "active") == 0)
      {
        if (!is_true && !is_false)
          return fail(Status::ErrorInvalidConfig, "%s.active must be a boolean", parent);
        lp_.film_active[i] = is_true;
      }
      else if (std::strcmp(k, "shift") == 0)
      {
        if (!is_true && !is_false)
          return fail(Status::ErrorInvalidConfig, "%s.shift must be a boolean", parent);
        lp_.film_shift[i] = is_true;
      }
      else if (std::strcmp(k, "groups") == 0)
      {
        if (!as_int)
          return fail(Status::ErrorInvalidConfig, "%s.groups must be an integer", parent);
        if (v < 0 || v > 65535)
          return fail(Status::ErrorInvalidConfig, "%s.groups (%d) is out of range", parent, v);
        lp_.film_groups[i] = v;
      }
      return true;
    }
  }

  return true;
}

// =============================================================================
// Activations
// =============================================================================

inline void StreamConverter::beginActivation(bool secondary, bool is_array, uint16_t index)
{
  act_ = ActParse{};
  act_slot_ = secondary ? ActSlot::Secondary : ActSlot::Primary;
  act_in_array_ = is_array;
  act_index_ = index;

  if (secondary)
  {
    lp_.sec_seen = true;
    lp_.sec_array = is_array;
  }
  else
  {
    lp_.act_seen = true;
    lp_.act_array = is_array;
  }
}

inline bool StreamConverter::finishActivation()
{
  const unsigned la_n = static_cast<unsigned>(layer_array_count_);
  ActivationCfg cfg;

  if (!act_.type_set)
  {
    act_slot_ = ActSlot::None;
    act_obj_idx_ = stream_detail::kNoAnchor;
    return fail(Status::ErrorInvalidConfig, "layer array %u: activation object needs a string 'type'", la_n);
  }
  cfg.type = act_.type;

  // The parameter list is type-dependent and follows the trainer's from_json
  // exactly, including its asymmetry: a bare string "LeakyReLU" carries no
  // slope, while the object form defaults it to 0.01 and carries one.
  float params[8];
  size_t count = 0;

  if (act_.type == 4) // LeakyReLU
  {
    params[count++] = static_cast<float>(act_.leaky);
  }
  else if (act_.type == 5) // PReLU
  {
    if (act_.prelu_scalar_set)
      params[count++] = static_cast<float>(act_.prelu_scalar);
  }
  else if (act_.type == 9) // LeakyHardtanh
  {
    for (int i = 0; i < 4; i++)
      params[count++] = static_cast<float>(act_.lht[i]);
  }

  if (act_.type == 5 && act_.slopes_array && !act_.prelu_scalar_set)
  {
    // The per-channel slopes were streamed straight into the pool as they
    // arrived, so they are already contiguous at slopes_off.
    cfg.param_count = static_cast<uint8_t>(act_.slopes_n);
    cfg.param_offset = act_.slopes_off;
  }
  else if (count > 0)
  {
    uint16_t offset = 0;
    float* dst = scratch_.layer.claim(static_cast<uint8_t>(count), offset);
    if (dst == nullptr)
    {
      act_slot_ = ActSlot::None;
      act_obj_idx_ = stream_detail::kNoAnchor;
      return fail(Status::ErrorTooSmall, "layer array %u needs more than %u activation parameters", la_n,
                  static_cast<unsigned>(kActivationParamPool));
    }
    for (size_t i = 0; i < count; i++)
      dst[i] = params[i];
    cfg.param_count = static_cast<uint8_t>(count);
    cfg.param_offset = offset;
  }

  const bool secondary = (act_slot_ == ActSlot::Secondary);
  if (act_in_array_)
  {
    if (act_index_ < kMaxDilations)
    {
      if (secondary)
      {
        scratch_.layer.secondary[act_index_] = cfg;
        lp_.sec_kind[act_index_] = 2;
        lp_.sec_count = static_cast<uint16_t>(act_index_ + 1);
      }
      else
      {
        scratch_.layer.activations[act_index_] = cfg;
        lp_.act_count = static_cast<uint16_t>(act_index_ + 1);
      }
    }
  }
  else if (secondary)
  {
    lp_.sec_single = cfg;
    lp_.sec_single_kind = 2;
  }
  else
  {
    lp_.act_single = cfg;
  }

  act_slot_ = ActSlot::None;
  act_obj_idx_ = stream_detail::kNoAnchor;
  return true;
}

inline bool StreamConverter::routeActivationValue(lwjson_stream_type_t type)
{
  const size_t a = act_obj_idx_;
  if (a == stream_detail::kNoAnchor)
    return false;

  // Members of the activation object.
  if (depth() == a + 2)
  {
    const char* k = keyAt(a + 1);
    if (k == nullptr)
      return false;

    if (std::strcmp(k, "type") == 0)
    {
      if (type != LWJSON_STREAM_TYPE_STRING)
        return fail(Status::ErrorInvalidConfig, "activation object needs a string 'type'");
      if (!stream_detail::ActivationTypeFromName(str(), act_.type))
        return fail(Status::ErrorInvalidConfig, "unknown activation '%s'", str());
      act_.type_set = true;
      return true;
    }

    if (type != LWJSON_STREAM_TYPE_NUMBER)
      return true;

    if (std::strcmp(k, "negative_slope") == 0)
    {
      const double v = stream_detail::TokenToF64(prim());
      act_.leaky = v;
      act_.leaky_set = true;
      act_.prelu_scalar = v;
      act_.prelu_scalar_set = true;
      return true;
    }

    static const char* const kLht[4] = {"min_val", "max_val", "min_slope", "max_slope"};
    for (int i = 0; i < 4; i++)
    {
      if (std::strcmp(k, kLht[i]) == 0)
      {
        act_.lht[i] = stream_detail::TokenToF64(prim());
        act_.lht_set[i] = true;
        return true;
      }
    }
    return true;
  }

  // negative_slopes[] -- streamed into the parameter pool as they arrive, so a
  // per-channel PReLU never needs a buffer of its own.
  if (depth() == a + 3 && isArray(a + 2) && isKey(a + 1, "negative_slopes"))
  {
    if (type != LWJSON_STREAM_TYPE_NUMBER)
      return fail(Status::ErrorInvalidConfig, "negative_slopes must all be numbers");
    if (act_.slopes_n >= 255)
      return fail(Status::ErrorInvalidConfig, "more than the 255 negative_slopes the format allows");

    uint16_t offset = 0;
    float* p = scratch_.layer.claim(1, offset);
    if (p == nullptr)
      return fail(Status::ErrorTooSmall, "layer array needs more than %u activation parameters",
                  static_cast<unsigned>(kActivationParamPool));
    if (act_.slopes_n == 0)
      act_.slopes_off = offset;
    *p = static_cast<float>(stream_detail::TokenToF64(prim()));
    act_.slopes_n++;
    return true;
  }

  return false;
}

// =============================================================================
// Layer arrays
// =============================================================================

inline void StreamConverter::beginLayer()
{
  lp_ = LayerParse{};
  for (size_t i = 0; i < kNumFilmBlocks; i++)
  {
    lp_.film_active[i] = true;
    lp_.film_shift[i] = true;
    lp_.film_groups[i] = 1;
  }
  scratch_.layer.reset();
}

inline bool StreamConverter::finalizeLayer()
{
  Cfg& out = scratch_.layer;
  const unsigned la_n = static_cast<unsigned>(layer_array_count_);

  const size_t n = lp_.dilations;
  if (n == 0)
    return fail(Status::ErrorInvalidConfig, "layer array %u: 'dilations' must be a non-empty array", la_n);
  out.num_dilations = static_cast<uint8_t>(n);

  // --- required scalars ---
  struct Req
  {
    const char* key;
    bool set;
    int value;
  };
  const Req required[3] = {
    {"input_size", lp_.input_size_set, lp_.input_size},
    {"condition_size", lp_.condition_size_set, lp_.condition_size},
    {"channels", lp_.channels_set, lp_.channels},
  };
  for (const Req& r : required)
  {
    if (!r.set || r.value < 0 || r.value > 65535)
      return fail(Status::ErrorInvalidConfig, "layer array %u: '%s' must be an integer in 0..65535", la_n, r.key);
  }

  out.input_size = static_cast<uint16_t>(lp_.input_size);
  out.condition_size = static_cast<uint16_t>(lp_.condition_size);
  out.channels = static_cast<uint16_t>(lp_.channels);
  out.bottleneck = static_cast<uint16_t>(lp_.bottleneck_set ? lp_.bottleneck : lp_.channels);
  out.groups_input = static_cast<uint16_t>(lp_.groups_input);
  out.groups_input_mixin = static_cast<uint16_t>(lp_.groups_input_mixin);

  // --- head ---
  // The trainer nests these under "head"; older files carry head_size and
  // head_bias at the layer-array level with an implicit kernel size and
  // dilation of 1.
  int head_size = 0;
  int head_kernel = 1;
  int head_dilation = 1;
  bool head_bias = false;

  if (lp_.head_obj)
  {
    if (!lp_.head_out_channels_set || !lp_.head_kernel_set || !lp_.head_bias_set)
      return fail(Status::ErrorInvalidConfig, "layer array %u: head needs out_channels, kernel_size and bias", la_n);
    head_size = lp_.head_out_channels;
    head_kernel = lp_.head_kernel;
    head_dilation = lp_.head_dilation;
    head_bias = lp_.head_bias;
  }
  else if (lp_.legacy_head_size_set)
  {
    if (!lp_.legacy_head_bias_set)
      return fail(Status::ErrorInvalidConfig, "layer array %u: legacy head_size/head_bias are malformed", la_n);
    head_size = lp_.legacy_head_size;
    head_bias = lp_.legacy_head_bias;
  }
  else
  {
    return fail(Status::ErrorInvalidConfig,
                "layer array %u: expected a 'head' object, or legacy head_size and head_bias", la_n);
  }

  if (head_kernel < 1)
    return fail(Status::ErrorInvalidConfig, "layer array %u: head.kernel_size must be >= 1", la_n);
  if (head_size < 0 || head_size > 65535 || head_kernel > 65535)
    return fail(Status::ErrorInvalidConfig, "layer array %u: head size or kernel size is out of range", la_n);

  out.head_size = static_cast<uint16_t>(head_size);
  out.head_kernel_size = static_cast<uint16_t>(head_kernel);
  out.head_dilation = static_cast<int32_t>(head_dilation);
  out.head_bias = head_bias ? 1 : 0;

  // --- kernel sizes: exactly one of the scalar or the array ---
  if (lp_.ks_scalar_set && lp_.ks_array)
    return fail(Status::ErrorInvalidConfig, "layer array %u: give kernel_size or kernel_sizes, not both", la_n);

  if (lp_.ks_array)
  {
    if (lp_.ks_count != n)
      return fail(Status::ErrorInvalidConfig, "layer array %u: %u kernel_sizes for %u dilations", la_n,
                  static_cast<unsigned>(lp_.ks_count), static_cast<unsigned>(n));
  }
  else if (lp_.ks_scalar_set)
  {
    for (size_t i = 0; i < n; i++)
      out.kernel_sizes[i] = static_cast<int32_t>(lp_.ks_scalar);
  }
  else
  {
    return fail(Status::ErrorInvalidConfig, "layer array %u: needs kernel_size or kernel_sizes", la_n);
  }

  // --- 1x1 blocks ---
  out.layer1x1_active = lp_.layer1x1_active ? 1 : 0;
  out.layer1x1_groups = static_cast<uint16_t>(lp_.layer1x1_groups);
  out.head1x1_active = lp_.head1x1_active ? 1 : 0;
  out.head1x1_out_channels = static_cast<uint16_t>(lp_.head1x1_out_set ? lp_.head1x1_out : lp_.channels);
  out.head1x1_groups = static_cast<uint16_t>(lp_.head1x1_groups);

  // --- FiLM ---
  for (size_t i = 0; i < kNumFilmBlocks; i++)
  {
    FilmCfg& f = out.film[i];
    f = FilmCfg{}; // absent: flags 0, groups 1

    if (!lp_.film_seen[i])
      continue;

    if (lp_.film_is_bool[i])
    {
      if (lp_.film_bool[i])
        f.flags = 0x01 | 0x02;
      continue; // false is absent
    }

    uint8_t flags = 0;
    if (lp_.film_active[i])
      flags |= 0x01;
    if (lp_.film_shift[i])
      flags |= 0x02;
    f.flags = flags;
    f.groups = static_cast<uint16_t>(lp_.film_groups[i]);
  }

  // --- activations ---
  if (!lp_.act_seen)
    return fail(Status::ErrorInvalidConfig, "layer array %u: 'activation' is missing", la_n);

  if (lp_.act_array)
  {
    if (lp_.act_count != n)
      return fail(Status::ErrorInvalidConfig, "layer array %u: %u activations for %u dilations", la_n,
                  static_cast<unsigned>(lp_.act_count), static_cast<unsigned>(n));
  }
  else
  {
    // A single activation broadcasts across the array. One pool entry serves
    // all of them: the emitted bytes are what has to match, not the layout.
    for (size_t i = 0; i < n; i++)
      out.activations[i] = lp_.act_single;
  }

  // --- gating ---
  switch (lp_.gating)
  {
    case LayerParse::Gating::Array:
      if (lp_.gating_count != n)
        return fail(Status::ErrorInvalidConfig, "layer array %u: %u gating modes for %u dilations", la_n,
                    static_cast<unsigned>(lp_.gating_count), static_cast<unsigned>(n));
      break;
    case LayerParse::Gating::Scalar:
    case LayerParse::Gating::Legacy: std::memset(out.gating, lp_.gating_scalar, n); break;
    case LayerParse::Gating::Absent:
    default: std::memset(out.gating, GATING_NONE, n); break;
  }

  // --- secondary activations ---
  // One per dilation whatever the gating, so the block is a fixed shape; the
  // loader only reads the ones whose gating mode is not NONE, which is why a
  // null in the source is only a problem where it would actually be read.
  for (size_t i = 0; i < n; i++)
  {
    ActivationCfg& dst = out.secondary[i];

    if (out.gating[i] == GATING_NONE)
    {
      dst = ActivationCfg{};
      dst.type = stream_detail::kActivationTanh; // placeholder; never read
      continue;
    }

    if (!lp_.sec_seen)
    {
      dst = ActivationCfg{};
      dst.type = stream_detail::kActivationSigmoid; // the gate's default
      continue;
    }

    if (lp_.sec_array)
    {
      if (i >= lp_.sec_count)
        return fail(Status::ErrorInvalidConfig, "layer array %u: %u secondary activations for %u dilations", la_n,
                    static_cast<unsigned>(lp_.sec_count), static_cast<unsigned>(n));
      if (lp_.sec_kind[i] != 2)
        return fail(Status::ErrorInvalidConfig,
                    "layer array %u secondary %u: activation must be a string or an object", la_n,
                    static_cast<unsigned>(i));
    }
    else
    {
      if (lp_.sec_single_kind != 2)
        return fail(Status::ErrorInvalidConfig,
                    "layer array %u secondary %u: activation must be a string or an object", la_n,
                    static_cast<unsigned>(i));
      dst = lp_.sec_single;
    }
  }

  return true;
}

inline void StreamConverter::endLayer()
{
  // Decide before validating, not after. Whether this is the submodel asked for
  // turns on its channel count alone, and a container's other half should not
  // have to be well-formed for us to say it is not the one -- otherwise "no
  // channels=3 submodel here" comes back as a complaint about the channels=8
  // one's dilations.
  if (!decided_ && !decideOnFirstLayer())
    return; // a submodel we did not want, or a failure already recorded

  if (skip_model_ || !emitting_)
    return;

  if (!finalizeLayer())
    return;

  if (layer_array_count_ == 255)
  {
    fail(Status::ErrorInvalidConfig, "more than the 255 layer arrays the format allows");
    return;
  }

  EmitLayerArray(w_, scratch_.layer);
  layer_array_count_++;
}

// =============================================================================
// finish()
// =============================================================================

inline Status StreamConverter::finish()
{
  if (failed())
    return status_;

  if (!emitting_)
  {
    return FailImpl(result_->detail, Status::ErrorInvalidConfig,
                    "source ended before a convertible model was found");
  }

  if (!block_closed_)
  {
    return FailImpl(result_->detail, Status::ErrorTruncated, "source ended inside the model configuration");
  }

  if (w_.failed())
  {
    return FailImpl(result_->detail, Status::ErrorTooSmall,
                    "output buffer of %u bytes is too small for this model",
                    static_cast<unsigned>(capacity_));
  }

  // ---- metadata block, into the 48 bytes reserved at offset 32 ----
  MetadataCfg meta;
  if (!version_set_)
    return FailImpl(result_->detail, Status::ErrorInvalidConfig, "model has no string 'version'");
  stream_detail::ParseVersion(version_, meta.version);

  // The fold: the submodel's own value wins, the container's fills the gap.
  // loudness lives on both and belongs to the weights that produced it;
  // input_level_dbu and output_level_dbu are only ever on the container, and
  // without the fallback a submodel would lose two of the three level fields
  // .namb can carry.
  static const uint8_t kFlags[3] = {META_HAS_LOUDNESS, META_HAS_INPUT_LEVEL, META_HAS_OUTPUT_LEVEL};
  double* const dest[3] = {&meta.loudness, &meta.input_level, &meta.output_level};
  for (int i = 0; i < 3; i++)
  {
    if (inner_.has[i])
    {
      *dest[i] = inner_.val[i];
      meta.flags |= kFlags[i];
    }
    else if (is_container_ && outer_.has[i])
    {
      *dest[i] = outer_.val[i];
      meta.flags |= kFlags[i];
    }
  }
  meta.sample_rate = sample_rate_;

  SpanWriter mw(out_ + FILE_HEADER_SIZE, METADATA_BLOCK_SIZE);
  EmitMetadataBlock(mw, meta);

  // ---- header ----
  const uint32_t total_size = static_cast<uint32_t>(w_.position());
  FinalizeFileHeader(w_, header_patch_, total_size, static_cast<uint32_t>(weights_offset_), weight_count_,
                     static_cast<uint32_t>(model_block_size_), out_);

  result_->size = total_size;
  result_->weight_count = weight_count_;

  // ---- name ----
  // The container carries the human name; a submodel's metadata has only date,
  // loudness and gain. Outer first, therefore -- the reverse of the level
  // fields above.
  const char* name = nullptr;
  if (is_container_ && outer_.name_set)
    name = outer_.name;
  else if (inner_.name_set)
    name = inner_.name;

  if (name != nullptr)
  {
    SanitizeName(name, result_->name, kNameSize);
    if (result_->name[0] != '\0')
      return Status::Ok;
  }
  // Sanitising can empty a name that was all punctuation, so the fallback is
  // tried after that too, not only when the key is missing.
  if (opts_.fallback_name != nullptr)
  {
    SanitizeName(opts_.fallback_name, result_->name, kNameSize);
    if (result_->name[0] != '\0')
      return Status::Ok;
  }
  SanitizeName("model", result_->name, kNameSize);
  return Status::Ok;
}

// =============================================================================
// Free function
// =============================================================================

/// \brief Convert a .nam read from \p src into a .namb blob.
///
/// \param src      Byte source; see the SOURCE CONTRACT note at the top.
/// \param out      Destination span. 16 KiB is comfortable for A2-Lite, whose
///                 blob is a little over 8 KiB.
/// \param capacity Bytes available at \p out.
/// \param result   Size, weight count and suggested name on success; \p detail
///                 carries the reason on failure.
/// \param opts     Which submodel to take out of a container.
/// \param conv     Caller-owned working state, ~16 KB. Reusable across models.
template<class Source>
Status WriteNambStream(Source& src, uint8_t* out, size_t capacity, WriteResult& result, const WriteOptions& opts,
                       StreamConverter& conv)
{
  const Status started = conv.begin(out, capacity, opts, result);
  if (!IsOk(started))
    return started;

  for (;;)
  {
    const int c = src.get();
    if (c < 0)
      break;
    const Status st = conv.feed(static_cast<char>(c));
    if (!IsOk(st))
      return st;
    // The chosen model is fully read well before the file ends -- a channels=3
    // import stops before the channels=8 weights it would otherwise stream past.
    if (conv.complete())
      break;
  }

  return conv.finish();
}

} // namespace namb
} // namespace nam
