#pragma once
// .nam (JSON) -> .namb (binary) conversion, as a library rather than a program.
//
// Generic and embedded-friendly namb writer with changes from nam2namb:
//
//   No exceptions. The firmware builds -fno-exceptions, under which nlohmann's
//   JSON_THROW becomes std::abort() -- so a truncated model on an SD card would
//   reset the pedal. Nothing here uses at(), operator[] on a maybe-absent key,
//   or a bare get<T>(); every read goes through the checked accessors below and
//   every failure comes back as a Status plus a sentence saying what was wrong.
//
//   No allocation. Output goes to a caller-supplied span, and weights are
//   streamed rather than collected, so converting a model costs no heap at all
//   beyond whatever the caller's JSON DOM already occupies.
//
//   A template on the JSON type. The firmware parses into nam::json_arena::
//   arena_json, whose allocator draws from a bump arena; the host tools use
//   nlohmann::json. Those are unrelated types, so the writer is generic over
//   both rather than picking one.
//
// Containers are handled here too. Every A2 model ships as a SlimmableContainer
// wrapping A2-Lite (channels=3) and A2-Full (channels=8), and .namb has no
// container architecture ID -- so rather than splitting to an intermediate file
// the way the host pipeline does, WriteNamb() walks config.submodels[], takes
// the one with the requested channel count, and converts that. The container's
// metadata is folded in as it goes: input_level_dbu and output_level_dbu live
// only on the outer document, and a submodel converted without them loses two
// of the three level fields .namb can carry.
//
//   uint8_t blob[16 * 1024];
//   nam::namb::WriteResult r;
//   nam::Status st = nam::namb::WriteNamb(doc, blob, sizeof(blob), r);
//   if (!nam::IsOk(st)) { report(r.detail); }
//   else                { commit(blob, r.size, r.name); }
//
// Only WaveNet and SlimmableContainer are understood. The other architectures
// .namb can encode are not ones this product loads -- get_dsp_namb is built
// NAMB_WAVENET_ONLY -- and a converter that emits models the loader rejects is
// worse than one that says so up front.

#include <cstdarg>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include <NAM/status.h>

#include "namb_format.h"

namespace nam
{
namespace namb
{

/// \brief Characters reserved for the failure sentence, including the NUL.
static constexpr size_t kDetailSize = 160;

/// \brief Characters reserved for the suggested pack entry name. Matches
///        nambpack::NAME_SIZE so a result can be handed straight to a pack
///        writer without a second truncation decision.
static constexpr size_t kNameSize = 32;

/// \brief What to convert out of the document.
struct WriteOptions
{
  /// Which slimmable submodel to take, by its layer-array channel count.
  /// Ignored for a document that is already a bare WaveNet.
  uint16_t channels = 3;

  /// Name to fall back on when the document carries none of its own.
  ///
  /// Not hypothetical: of the four shipping models, Deluxe_Reverb.nam has no
  /// metadata.name at either level. The source filename is the obvious thing
  /// to pass, and only the caller can see it.
  const char* fallback_name = nullptr;
};

/// \brief What came out, or why nothing did.
struct WriteResult
{
  size_t size = 0; ///< Bytes written to the output span
  uint16_t channels = 0; ///< Channel count of the submodel actually converted
  uint32_t weight_count = 0; ///< Weights in the emitted blob
  char name[kNameSize] = {}; ///< Suggested pack entry name, NUL-terminated
  char detail[kDetailSize] = {}; ///< Failure reason, or "" on success
};

// =============================================================================
// SpanWriter
// =============================================================================

/// \brief Bounds-checked writer over a fixed span.
///
/// Overflow is sticky rather than thrown, mirroring BinaryReader: the caller
/// checks once at the end instead of guarding every field, and a run past the
/// end produces a well-defined short write that is then rejected wholesale
/// rather than a buffer overrun.
class SpanWriter
{
public:
  SpanWriter(uint8_t* data, size_t capacity)
  : _data(data)
  , _capacity(data == nullptr ? 0 : capacity)
  {
  }

  bool failed() const { return _failed; }
  size_t position() const { return _pos; }
  uint8_t* data() { return _data; }

  void write_u8(uint8_t v)
  {
    if (check(1))
      _data[_pos++] = v;
  }

  void write_u16(uint16_t v) { write_raw(&v, 2); }
  void write_u32(uint32_t v) { write_raw(&v, 4); }
  void write_i32(int32_t v) { write_raw(&v, 4); }
  void write_f32(float v) { write_raw(&v, 4); }
  void write_f64(double v) { write_raw(&v, 8); }

  void write_zeros(size_t n)
  {
    if (check(n))
    {
      std::memset(_data + _pos, 0, n);
      _pos += n;
    }
  }

  /// \brief Pad to the next multiple of \p align.
  void align_to(size_t align)
  {
    while ((_pos % align) != 0)
      write_u8(0);
  }

  /// \brief Backpatch a uint32 already written at \p offset. Silently ignored
  ///        once the writer has overflowed, since the buffer is being discarded
  ///        anyway and \p offset may no longer mean anything.
  void set_u32(size_t offset, uint32_t v)
  {
    if (_failed || offset + 4 > _pos)
      return;
    std::memcpy(_data + offset, &v, 4);
  }

private:
  template<class T>
  void write_raw(const T* v, size_t n)
  {
    if (check(n))
    {
      std::memcpy(_data + _pos, v, n);
      _pos += n;
    }
  }

  bool check(size_t n)
  {
    if (_failed)
      return false;
    if (_pos + n > _capacity)
    {
      _failed = true;
      return false;
    }
    return true;
  }

  uint8_t* _data;
  size_t _capacity;
  size_t _pos = 0;
  bool _failed = false;
};

namespace writer_detail
{

// =============================================================================
// Checked accessors
// =============================================================================
//
// Every one of these answers "is it there, and is it the type I need" in a
// single call that cannot throw. A null member is treated as an absent one
// throughout: the trainer emits "head": null for a layer array with no head,
// and a caller that distinguished the two would have to check both everywhere.

template<class J>
const J* Member(const J& o, const char* key)
{
  if (!o.is_object())
    return nullptr;
  const auto it = o.find(key);
  if (it == o.end() || it->is_null())
    return nullptr;
  return &(*it);
}

template<class J>
bool AsInt(const J& v, int& out)
{
  if (v.is_number_integer() || v.is_number_unsigned())
  {
    out = static_cast<int>(v.template get<int64_t>());
    return true;
  }
  // Exporters sometimes write a whole number as a float. Accept that, but not
  // a fractional one -- silently truncating a dilation would be worse than
  // refusing the file.
  if (v.is_number_float())
  {
    const double d = v.template get<double>();
    const double t = static_cast<double>(static_cast<int64_t>(d));
    if (d == t)
    {
      out = static_cast<int>(static_cast<int64_t>(d));
      return true;
    }
  }
  return false;
}

template<class J>
bool AsDouble(const J& v, double& out)
{
  if (!v.is_number())
    return false;
  out = v.template get<double>();
  return true;
}

template<class J>
bool AsBool(const J& v, bool& out)
{
  if (!v.is_boolean())
    return false;
  out = v.template get<bool>();
  return true;
}

/// \brief Borrowed pointer to a string value, or nullptr. No copy, so this
///        costs nothing even for the arena DOM whose strings are on the heap.
template<class J>
const typename J::string_t* AsString(const J& v)
{
  if (!v.is_string())
    return nullptr;
  return v.template get_ptr<const typename J::string_t*>();
}

/// \brief Read an integer member, or leave \p out alone if it is absent.
/// \return false only if the member is present but not an integer.
template<class J>
bool IntOr(const J& o, const char* key, int& out)
{
  const J* m = Member(o, key);
  if (m == nullptr)
    return true;
  return AsInt(*m, out);
}

template<class J>
bool BoolOr(const J& o, const char* key, bool& out)
{
  const J* m = Member(o, key);
  if (m == nullptr)
    return true;
  return AsBool(*m, out);
}

// =============================================================================
// Conversion context
// =============================================================================

/// Carries the output span and the diagnostic buffer through the recursion, so
/// no function has to thread three parameters it does not use.
template<class J>
struct Context
{
  SpanWriter& w;
  char* detail;
};

// Records why the conversion stopped and returns the status, so a failure is
// never reported as a bare enum the operator cannot act on.
inline Status FailImpl(char* detail, Status status, const char* fmt, ...)
{
  va_list args;
  va_start(args, fmt);
  std::vsnprintf(detail, kDetailSize, fmt, args);
  va_end(args);
  return status;
}

#define NAMB_FAIL(ctx, status, ...) return ::nam::namb::writer_detail::FailImpl((ctx).detail, status, __VA_ARGS__)

// =============================================================================
// Activation configs
// =============================================================================

/// Mirrors activations::ActivationConfig::from_json, which cannot be reused:
/// it is declared against nlohmann::json specifically, and the firmware parses
/// into a different basic_json instantiation. The mapping is duplicated rather
/// than the file being templated, because the .namb encoding only needs the
/// type id and the parameter list -- not the config object.
inline bool ActivationTypeFromName(const char* name, uint8_t& out)
{
  // Order matches nam::activations::ActivationType. The two LeakyHardtanh
  // spellings are both accepted upstream, so both are accepted here.
  struct Entry
  {
    const char* name;
    uint8_t id;
  };
  static const Entry kTable[] = {
    {"Tanh", 0},     {"Hardtanh", 1},      {"Fasttanh", 2},      {"ReLU", 3},
    {"LeakyReLU", 4}, {"PReLU", 5},        {"Sigmoid", 6},       {"SiLU", 7},
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

static constexpr uint8_t kActivationSigmoid = 6;
static constexpr uint8_t kActivationTanh = 0;

/// Writes one activation config: type byte, parameter count, then the floats.
///
/// The parameter list is type-dependent and follows from_json exactly,
/// including its asymmetry: a bare string "LeakyReLU" carries no slope and so
/// emits zero parameters, while the object form defaults the slope to 0.01 and
/// emits one. Reproducing that is what keeps this byte-identical to nam2namb.
template<class J>
Status WriteActivation(Context<J>& ctx, const J& act, const char* where)
{
  uint8_t type = 0;

  if (const typename J::string_t* s = AsString(act))
  {
    if (!ActivationTypeFromName(s->c_str(), type))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: unknown activation '%s'", where, s->c_str());
    ctx.w.write_u8(type);
    ctx.w.write_u8(0); // string form carries no parameters
    return Status::Ok;
  }

  if (!act.is_object())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: activation must be a string or an object", where);

  const J* type_node = Member(act, "type");
  const typename J::string_t* type_name = (type_node != nullptr) ? AsString(*type_node) : nullptr;
  if (type_name == nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: activation object needs a string 'type'", where);
  if (!ActivationTypeFromName(type_name->c_str(), type))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: unknown activation '%s'", where, type_name->c_str());

  ctx.w.write_u8(type);

  // --- gather parameters ----------------------------------------------------
  float params[8];
  size_t count = 0;
  const J* slopes = nullptr; // PReLU per-channel, written straight from the DOM

  if (type == 4) // LeakyReLU
  {
    double slope = 0.01;
    if (const J* m = Member(act, "negative_slope"))
    {
      if (!AsDouble(*m, slope))
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: negative_slope must be a number", where);
    }
    params[count++] = static_cast<float>(slope);
  }
  else if (type == 5) // PReLU
  {
    if (const J* m = Member(act, "negative_slope"))
    {
      double slope = 0.0;
      if (!AsDouble(*m, slope))
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: negative_slope must be a number", where);
      params[count++] = static_cast<float>(slope);
    }
    else if (const J* arr = Member(act, "negative_slopes"))
    {
      if (!arr->is_array())
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: negative_slopes must be an array", where);
      if (arr->size() > 255)
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: %u slopes exceeds the 255 the format allows", where,
                  static_cast<unsigned>(arr->size()));
      slopes = arr;
    }
  }
  else if (type == 9) // LeakyHardtanh
  {
    static const char* kKeys[4] = {"min_val", "max_val", "min_slope", "max_slope"};
    static const double kDefaults[4] = {-1.0, 1.0, 0.01, 0.01};
    for (int i = 0; i < 4; i++)
    {
      double v = kDefaults[i];
      if (const J* m = Member(act, kKeys[i]))
      {
        if (!AsDouble(*m, v))
          NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: %s must be a number", where, kKeys[i]);
      }
      params[count++] = static_cast<float>(v);
    }
  }

  // --- emit -----------------------------------------------------------------
  if (slopes != nullptr)
  {
    ctx.w.write_u8(static_cast<uint8_t>(slopes->size()));
    for (const auto& s : *slopes)
    {
      double v = 0.0;
      if (!AsDouble(s, v))
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s: negative_slopes must all be numbers", where);
      ctx.w.write_f32(static_cast<float>(v));
    }
  }
  else
  {
    ctx.w.write_u8(static_cast<uint8_t>(count));
    for (size_t i = 0; i < count; i++)
      ctx.w.write_f32(params[i]);
  }

  return Status::Ok;
}

// =============================================================================
// FiLM parameters (4 bytes)
// =============================================================================

/// An ABSENT film block and a film block whose "active" is false are not the
/// same four bytes: absent writes flags 0, while {"active":false,"shift":true}
/// writes flags 0x02. nam2namb has always drawn that distinction and the
/// loader reads shift independently of active, so it is preserved here.
template<class J>
Status WriteFilm(Context<J>& ctx, const J& layer, const char* key)
{
  const J* film = Member(layer, key);

  bool present = (film != nullptr);
  if (present && film->is_boolean())
  {
    // "conv_pre_film": false is shorthand for absent. The true spelling has
    // never appeared in a model, but reading it as "active with defaults" is
    // the only sensible meaning if it ever does.
    bool on = false;
    (void)AsBool(*film, on);
    if (!on)
      present = false;
    else
    {
      ctx.w.write_u8(0x01 | 0x02);
      ctx.w.write_u8(0);
      ctx.w.write_u16(1);
      return Status::Ok;
    }
  }

  if (!present)
  {
    ctx.w.write_u8(0); // flags: not active
    ctx.w.write_u8(0); // reserved
    ctx.w.write_u16(1); // groups
    return Status::Ok;
  }

  if (!film->is_object())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s must be an object or a boolean", key);

  bool active = true;
  bool shift = true;
  int groups = 1;
  if (!BoolOr(*film, "active", active))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s.active must be a boolean", key);
  if (!BoolOr(*film, "shift", shift))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s.shift must be a boolean", key);
  if (!IntOr(*film, "groups", groups))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s.groups must be an integer", key);
  if (groups < 0 || groups > 65535)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%s.groups (%d) is out of range", key, groups);

  uint8_t flags = 0;
  if (active)
    flags |= 0x01;
  if (shift)
    flags |= 0x02;

  ctx.w.write_u8(flags);
  ctx.w.write_u8(0);
  ctx.w.write_u16(static_cast<uint16_t>(groups));
  return Status::Ok;
}

// =============================================================================
// Gating
// =============================================================================

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

/// Resolves the per-layer gating modes once, into \p out. The modes are needed
/// twice -- written as a block, then again to decide which layers get a
/// secondary activation -- and nam2namb parsed them twice to do it. Doing it
/// once removes the chance of the two passes disagreeing.
template<class J>
Status ResolveGating(Context<J>& ctx, const J& layer, size_t n, uint8_t* out, size_t la)
{
  if (const J* gm = Member(layer, "gating_mode"))
  {
    if (gm->is_array())
    {
      if (gm->size() != n)
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: %u gating modes for %u dilations",
                  static_cast<unsigned>(la), static_cast<unsigned>(gm->size()), static_cast<unsigned>(n));
      size_t i = 0;
      for (const auto& v : *gm)
      {
        const typename J::string_t* s = AsString(v);
        if (s == nullptr || !GatingModeFromName(s->c_str(), out[i]))
          NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: bad gating_mode at index %u",
                    static_cast<unsigned>(la), static_cast<unsigned>(i));
        i++;
      }
      return Status::Ok;
    }

    const typename J::string_t* s = AsString(*gm);
    uint8_t mode = GATING_NONE;
    if (s == nullptr || !GatingModeFromName(s->c_str(), mode))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: bad gating_mode", static_cast<unsigned>(la));
    std::memset(out, mode, n);
    return Status::Ok;
  }

  if (const J* gated = Member(layer, "gated")) // pre-gating_mode files
  {
    bool on = false;
    if (!AsBool(*gated, on))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: 'gated' must be a boolean",
                static_cast<unsigned>(la));
    std::memset(out, on ? GATING_GATED : GATING_NONE, n);
    return Status::Ok;
  }

  std::memset(out, GATING_NONE, n);
  return Status::Ok;
}

// =============================================================================
// Metadata block (48 bytes)
// =============================================================================

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

/// Looks a metadata field up in the submodel first and the container second.
///
/// This is what folding a container's metadata down amounts to, for the three
/// fields .namb stores. loudness appears on both and the submodel's belongs to
/// its weights, so it wins; input_level_dbu and output_level_dbu appear only on
/// the container, so without the fallback they would be lost. Same precedence
/// split_slimmable applies when it writes an intermediate file.
template<class J>
const J* MetaField(const J* inner, const J* outer, const char* key)
{
  if (inner != nullptr)
  {
    if (const J* v = Member(*inner, key))
      return v;
  }
  if (outer != nullptr)
  {
    if (const J* v = Member(*outer, key))
      return v;
  }
  return nullptr;
}

template<class J>
Status WriteMetadataBlock(Context<J>& ctx, const J& model, const J* outer_meta)
{
  const J* version = Member(model, "version");
  const typename J::string_t* version_str = (version != nullptr) ? AsString(*version) : nullptr;
  if (version_str == nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "model has no string 'version'");

  uint8_t v[3];
  ParseVersion(version_str->c_str(), v);
  ctx.w.write_u8(v[0]);
  ctx.w.write_u8(v[1]);
  ctx.w.write_u8(v[2]);

  const J* inner_meta = Member(model, "metadata");

  uint8_t flags = 0;
  double loudness = 0.0;
  double input_level = 0.0;
  double output_level = 0.0;

  struct Field
  {
    const char* key;
    uint8_t flag;
    double* dest;
  };
  const Field fields[3] = {
    {"loudness", META_HAS_LOUDNESS, &loudness},
    {"input_level_dbu", META_HAS_INPUT_LEVEL, &input_level},
    {"output_level_dbu", META_HAS_OUTPUT_LEVEL, &output_level},
  };

  for (const Field& f : fields)
  {
    if (const J* m = MetaField(inner_meta, outer_meta, f.key))
    {
      if (!AsDouble(*m, *f.dest))
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "metadata.%s must be a number", f.key);
      flags |= f.flag;
    }
  }
  ctx.w.write_u8(flags);

  double sample_rate = -1.0;
  if (const J* sr = Member(model, "sample_rate"))
  {
    if (!AsDouble(*sr, sample_rate))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "sample_rate must be a number");
  }

  ctx.w.write_f64(sample_rate);
  ctx.w.write_f64(loudness);
  ctx.w.write_f64(input_level);
  ctx.w.write_f64(output_level);
  ctx.w.write_zeros(12); // reserved

  return Status::Ok;
}

// =============================================================================
// WaveNet layer array
// =============================================================================

/// Largest dilation count a layer array may declare. Not an arbitrary cap:
/// num_dilations is a uint8 in the container, so 255 is the format's own limit,
/// and the one buffer still held across the write is sized from it.
static constexpr size_t kMaxDilations = 255;

template<class J>
Status WriteLayerArray(Context<J>& ctx, const J& layer, size_t la)
{
  const unsigned la_n = static_cast<unsigned>(la);

  if (!layer.is_object())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u is not an object", la_n);

  const J* dilations = Member(layer, "dilations");
  if (dilations == nullptr || !dilations->is_array() || dilations->empty())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: 'dilations' must be a non-empty array", la_n);

  const size_t n = dilations->size();
  if (n > kMaxDilations)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: %u dilations exceeds the %u the format allows", la_n,
              static_cast<unsigned>(n), static_cast<unsigned>(kMaxDilations));

  // --- required scalars -----------------------------------------------------
  int input_size = 0;
  int condition_size = 0;
  int channels = 0;
  for (const char* key : {"input_size", "condition_size", "channels"})
  {
    const J* m = Member(layer, key);
    int value = 0;
    if (m == nullptr || !AsInt(*m, value) || value < 0 || value > 65535)
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: '%s' must be an integer in 0..65535", la_n, key);
    if (std::strcmp(key, "input_size") == 0)
      input_size = value;
    else if (std::strcmp(key, "condition_size") == 0)
      condition_size = value;
    else
      channels = value;
  }

  int bottleneck = channels;
  int groups_input = 1;
  int groups_input_mixin = 1;
  if (!IntOr(layer, "bottleneck", bottleneck) || !IntOr(layer, "groups_input", groups_input)
      || !IntOr(layer, "groups_input_mixin", groups_input_mixin))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: bottleneck/groups_* must be integers", la_n);

  // --- head -----------------------------------------------------------------
  // The trainer nests these under "head"; older files carry head_size and
  // head_bias at the layer-array level with an implicit kernel size and
  // dilation of 1. Mirrors parse_config_json().
  int head_size = 0;
  int head_kernel_size = 1;
  int head_dilation = 1;
  bool head_bias = false;

  if (const J* head = Member(layer, "head"))
  {
    if (!head->is_object())
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: 'head' must be an object", la_n);

    const J* out_channels = Member(*head, "out_channels");
    const J* kernel_size = Member(*head, "kernel_size");
    const J* bias = Member(*head, "bias");
    if (out_channels == nullptr || !AsInt(*out_channels, head_size) || kernel_size == nullptr
        || !AsInt(*kernel_size, head_kernel_size) || bias == nullptr || !AsBool(*bias, head_bias))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: head needs out_channels, kernel_size and bias",
                la_n);
    if (!IntOr(*head, "head_dilation", head_dilation))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: head.head_dilation must be an integer", la_n);
  }
  else if (const J* legacy = Member(layer, "head_size"))
  {
    const J* bias = Member(layer, "head_bias");
    if (!AsInt(*legacy, head_size) || bias == nullptr || !AsBool(*bias, head_bias))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: legacy head_size/head_bias are malformed", la_n);
  }
  else
  {
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig,
              "layer array %u: expected a 'head' object, or legacy head_size and head_bias", la_n);
  }

  if (head_kernel_size < 1)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: head.kernel_size must be >= 1", la_n);
  if (head_size < 0 || head_size > 65535 || head_kernel_size > 65535)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: head size or kernel size is out of range", la_n);

  // --- per-layer kernel sizes -----------------------------------------------
  // Exactly one of kernel_size (scalar, broadcast) or kernel_sizes (array).
  //
  // Validated here but not buffered: they are emitted further down, after the
  // dilations, and holding 255 of them across the intervening code would put a
  // kilobyte on the stack for the sake of one loop. The array is walked twice
  // instead -- once now to reject a bad one before anything is written, once
  // below to emit it.
  const J* ks_scalar = Member(layer, "kernel_size");
  const J* ks_array = Member(layer, "kernel_sizes");
  int ks_broadcast = 0;

  if (ks_scalar != nullptr && ks_array != nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: give kernel_size or kernel_sizes, not both", la_n);

  if (ks_array != nullptr)
  {
    if (!ks_array->is_array())
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: kernel_sizes must be an array", la_n);
    if (ks_array->size() != n)
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: %u kernel_sizes for %u dilations", la_n,
                static_cast<unsigned>(ks_array->size()), static_cast<unsigned>(n));
    size_t i = 0;
    for (const auto& k : *ks_array)
    {
      int value = 0;
      if (!AsInt(k, value))
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: kernel_sizes[%u] is not an integer", la_n,
                  static_cast<unsigned>(i));
      i++;
    }
  }
  else if (ks_scalar != nullptr)
  {
    if (!AsInt(*ks_scalar, ks_broadcast))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: kernel_size is not an integer", la_n);
  }
  else
  {
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: needs kernel_size or kernel_sizes", la_n);
  }

  // --- fixed block ----------------------------------------------------------
  ctx.w.write_u16(static_cast<uint16_t>(input_size));
  ctx.w.write_u16(static_cast<uint16_t>(condition_size));
  ctx.w.write_u16(static_cast<uint16_t>(head_size));
  ctx.w.write_u16(static_cast<uint16_t>(channels));
  ctx.w.write_u16(static_cast<uint16_t>(bottleneck));
  ctx.w.write_u16(static_cast<uint16_t>(head_kernel_size)); // v1 held a shared kernel_size here
  ctx.w.write_u8(head_bias ? 1 : 0);
  ctx.w.write_u8(static_cast<uint8_t>(n));
  ctx.w.write_u16(static_cast<uint16_t>(groups_input));
  ctx.w.write_u16(static_cast<uint16_t>(groups_input_mixin));
  ctx.w.write_i32(static_cast<int32_t>(head_dilation)); // v2

  // --- layer1x1 (4 bytes) ---------------------------------------------------
  bool layer1x1_active = true;
  int layer1x1_groups = 1;
  if (const J* l1 = Member(layer, "layer1x1"))
  {
    if (!l1->is_object() || !BoolOr(*l1, "active", layer1x1_active) || !IntOr(*l1, "groups", layer1x1_groups))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: layer1x1 is malformed", la_n);
  }
  ctx.w.write_u8(layer1x1_active ? 1 : 0);
  ctx.w.write_u16(static_cast<uint16_t>(layer1x1_groups));
  ctx.w.write_u8(0); // reserved

  // --- head1x1 (6 bytes) ----------------------------------------------------
  bool head1x1_active = false;
  int head1x1_out_channels = channels;
  int head1x1_groups = 1;
  if (const J* h1 = Member(layer, "head1x1"))
  {
    if (!h1->is_object() || !BoolOr(*h1, "active", head1x1_active)
        || !IntOr(*h1, "out_channels", head1x1_out_channels) || !IntOr(*h1, "groups", head1x1_groups))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: head1x1 is malformed", la_n);
  }
  ctx.w.write_u8(head1x1_active ? 1 : 0);
  ctx.w.write_u16(static_cast<uint16_t>(head1x1_out_channels));
  ctx.w.write_u16(static_cast<uint16_t>(head1x1_groups));
  ctx.w.write_u8(0); // reserved

  // --- 8 FiLM blocks (32 bytes) ---------------------------------------------
  static const char* kFilmKeys[8] = {
    "conv_pre_film",       "conv_post_film",       "input_mixin_pre_film", "input_mixin_post_film",
    "activation_pre_film", "activation_post_film", "layer1x1_post_film",   "head1x1_post_film",
  };
  for (const char* key : kFilmKeys)
  {
    const Status st = WriteFilm(ctx, layer, key);
    if (!IsOk(st))
      return st;
  }

  // --- dilations, then kernel sizes -----------------------------------------
  for (const auto& d : *dilations)
  {
    int value = 0;
    if (!AsInt(d, value))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: dilations must all be integers", la_n);
    ctx.w.write_i32(static_cast<int32_t>(value));
  }
  // v2. Second walk of what was validated above, so nothing had to be held.
  if (ks_array != nullptr)
  {
    for (const auto& k : *ks_array)
    {
      int value = 0;
      (void)AsInt(k, value); // already proven convertible
      ctx.w.write_i32(static_cast<int32_t>(value));
    }
  }
  else
  {
    for (size_t i = 0; i < n; i++)
      ctx.w.write_i32(static_cast<int32_t>(ks_broadcast));
  }

  // --- activations ----------------------------------------------------------
  const J* activation = Member(layer, "activation");
  if (activation == nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: 'activation' is missing", la_n);

  char where[48];
  if (activation->is_array())
  {
    if (activation->size() != n)
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: %u activations for %u dilations", la_n,
                static_cast<unsigned>(activation->size()), static_cast<unsigned>(n));
    size_t i = 0;
    for (const auto& a : *activation)
    {
      std::snprintf(where, sizeof(where), "layer array %u activation %u", la_n, static_cast<unsigned>(i++));
      const Status st = WriteActivation(ctx, a, where);
      if (!IsOk(st))
        return st;
    }
  }
  else
  {
    // A single activation broadcasts across the array.
    std::snprintf(where, sizeof(where), "layer array %u activation", la_n);
    for (size_t i = 0; i < n; i++)
    {
      const Status st = WriteActivation(ctx, *activation, where);
      if (!IsOk(st))
        return st;
    }
  }

  // --- gating modes ---------------------------------------------------------
  uint8_t gating[kMaxDilations];
  {
    const Status st = ResolveGating(ctx, layer, n, gating, la);
    if (!IsOk(st))
      return st;
  }
  for (size_t i = 0; i < n; i++)
    ctx.w.write_u8(gating[i]);

  // --- secondary activations ------------------------------------------------
  // One per layer whatever the gating, so the block is a fixed shape; the
  // loader only reads the ones whose gating mode is not NONE.
  const J* secondary = Member(layer, "secondary_activation");
  for (size_t i = 0; i < n; i++)
  {
    if (gating[i] == GATING_NONE)
    {
      ctx.w.write_u8(kActivationTanh); // placeholder; the type is never read
      ctx.w.write_u8(0);
      continue;
    }

    if (secondary == nullptr)
    {
      ctx.w.write_u8(kActivationSigmoid); // the gate's default
      ctx.w.write_u8(0);
      continue;
    }

    std::snprintf(where, sizeof(where), "layer array %u secondary %u", la_n, static_cast<unsigned>(i));
    if (secondary->is_array())
    {
      if (i >= secondary->size())
        NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "layer array %u: %u secondary activations for %u dilations", la_n,
                  static_cast<unsigned>(secondary->size()), static_cast<unsigned>(n));
      const Status st = WriteActivation(ctx, (*secondary)[i], where);
      if (!IsOk(st))
        return st;
    }
    else
    {
      const Status st = WriteActivation(ctx, *secondary, where);
      if (!IsOk(st))
        return st;
    }
  }

  return Status::Ok;
}

// =============================================================================
// Weights
// =============================================================================
//
// Counted in one pass and streamed in another, rather than collected into a
// vector. The two walks have to agree, so they are written as one traversal
// shape: condition_dsp first, then this model's own weights.

template<class J>
bool WeightsInOrder(const J& model, uint32_t& count, SpanWriter* w)
{
  const J* arch = Member(model, "architecture");
  const typename J::string_t* arch_name = (arch != nullptr) ? AsString(*arch) : nullptr;

  if (arch_name != nullptr && *arch_name == "WaveNet")
  {
    if (const J* config = Member(model, "config"))
    {
      if (const J* cdsp = Member(*config, "condition_dsp"))
      {
        if (!WeightsInOrder(*cdsp, count, w))
          return false;
      }
    }
  }

  if (const J* weights = Member(model, "weights"))
  {
    if (!weights->is_array())
      return false;
    for (const auto& v : *weights)
    {
      if (!v.is_number())
        return false;
      // get<float> rather than get<double> then a narrowing cast, so an
      // integer-typed weight converts exactly the way nam2namb converts it.
      if (w != nullptr)
        w->write_f32(v.template get<float>());
      count++;
    }
  }

  return true;
}

// =============================================================================
// Model block
// =============================================================================

template<class J>
Status WriteModelBlock(Context<J>& ctx, const J& model);

template<class J>
Status WriteWaveNetConfig(Context<J>& ctx, const J& model)
{
  const J* config = Member(model, "config");
  if (config == nullptr || !config->is_object())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "WaveNet model has no 'config' object");

  const J* layers = Member(*config, "layers");
  if (layers == nullptr || !layers->is_array() || layers->empty())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "WaveNet config needs a non-empty 'layers' array");
  if (layers->size() > 255)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "%u layer arrays exceeds the 255 the format allows",
              static_cast<unsigned>(layers->size()));

  int in_channels = 1;
  if (!IntOr(*config, "in_channels", in_channels))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "config.in_channels must be an integer");

  const J* head = Member(*config, "head");
  const J* cdsp = Member(*config, "condition_dsp");

  ctx.w.write_u8(static_cast<uint8_t>(in_channels));
  ctx.w.write_u8(head != nullptr ? 1 : 0);
  ctx.w.write_u8(static_cast<uint8_t>(layers->size()));
  ctx.w.write_u8(cdsp != nullptr ? 1 : 0);

  if (cdsp != nullptr)
  {
    uint32_t cdsp_weights = 0;
    if (!WeightsInOrder(*cdsp, cdsp_weights, nullptr))
      NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "condition_dsp weights are not a numeric array");
    ctx.w.write_u32(cdsp_weights);

    // Cast rather than a bare nullptr: J appears in the parameter type, and a
    // std::nullptr_t argument makes deduction fail even though J is already
    // fixed by the other arguments.
    const Status meta = WriteMetadataBlock(ctx, *cdsp, static_cast<const J*>(nullptr));
    if (!IsOk(meta))
      return meta;

    const Status block = WriteModelBlock(ctx, *cdsp);
    if (!IsOk(block))
      return block;
  }

  size_t la = 0;
  for (const auto& layer : *layers)
  {
    const Status st = WriteLayerArray(ctx, layer, la++);
    if (!IsOk(st))
      return st;
  }

  return Status::Ok;
}

template<class J>
Status WriteModelBlock(Context<J>& ctx, const J& model)
{
  const J* arch = Member(model, "architecture");
  const typename J::string_t* arch_name = (arch != nullptr) ? AsString(*arch) : nullptr;
  if (arch_name == nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "model has no string 'architecture'");

  if (*arch_name != "WaveNet")
  {
    NAMB_FAIL(ctx, Status::ErrorUnknownArchitecture,
              "architecture '%s' is not supported; this build converts WaveNet only", arch_name->c_str());
  }

  ctx.w.write_u8(ARCH_WAVENET);
  ctx.w.write_u8(0); // reserved

  const size_t config_size_offset = ctx.w.position();
  ctx.w.write_u16(0); // backpatched below
  const size_t config_start = ctx.w.position();

  const Status st = WriteWaveNetConfig(ctx, model);
  if (!IsOk(st))
    return st;

  const size_t config_size = ctx.w.position() - config_start;
  if (config_size > 65535)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "config block is %u bytes, too large for its uint16 length",
              static_cast<unsigned>(config_size));

  // set_u32 would clobber the two bytes after the field, so patch the uint16
  // directly. Safe: the writer has not failed, so config_size_offset is inside
  // what has been written.
  if (!ctx.w.failed())
  {
    const uint16_t cs = static_cast<uint16_t>(config_size);
    std::memcpy(ctx.w.data() + config_size_offset, &cs, 2);
  }

  return Status::Ok;
}

// =============================================================================
// Naming
// =============================================================================

/// Folds a model's display name down to something a pack entry can hold:
/// 31 characters of [A-Za-z0-9._-], with every run of anything else collapsed
/// to a single underscore and no underscore at either end.
///
/// Truncation can make two long names collide. That is the pack writer's
/// problem to resolve, not this function's -- it has no view of what is
/// already stored.
inline void SanitizeName(const char* src, char* dst, size_t dst_size)
{
  size_t out = 0;
  bool pending_sep = false;

  for (const char* p = src; *p != '\0' && out + 1 < dst_size; p++)
  {
    const char c = *p;
    const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') || c == '.'
                      || c == '_' || c == '-';
    if (keep)
    {
      if (pending_sep && out > 0 && out + 2 < dst_size)
        dst[out++] = '_';
      pending_sep = false;
      dst[out++] = c;
    }
    else if (out > 0)
    {
      pending_sep = true;
    }
  }

  while (out > 0 && dst[out - 1] == '_')
    out--;

  dst[out] = '\0';
}

template<class J>
void DeriveName(const J& model, const J* outer_meta, const char* fallback, char* dst, size_t dst_size)
{
  const J* inner_meta = Member(model, "metadata");
  // The container carries the human name; a submodel's metadata has only date,
  // loudness and gain. Outer first, therefore -- the reverse of the level
  // fields above.
  const J* name = nullptr;
  if (outer_meta != nullptr)
    name = Member(*outer_meta, "name");
  if (name == nullptr && inner_meta != nullptr)
    name = Member(*inner_meta, "name");

  const typename J::string_t* s = (name != nullptr) ? AsString(*name) : nullptr;
  if (s != nullptr)
  {
    SanitizeName(s->c_str(), dst, dst_size);
    if (dst[0] != '\0')
      return;
  }

  // Sanitising can empty a name that was all punctuation, so the fallback is
  // tried after that too, not only when the key is missing.
  if (fallback != nullptr)
  {
    SanitizeName(fallback, dst, dst_size);
    if (dst[0] != '\0')
      return;
  }

  SanitizeName("model", dst, dst_size);
}

// =============================================================================
// One model -> one blob
// =============================================================================

template<class J>
Status WriteOneModel(const J& model, const J* outer_meta, uint8_t* out, size_t capacity, WriteResult& result,
                     const WriteOptions& opts)
{
  SpanWriter w(out, capacity);
  Context<J> ctx{w, result.detail};

  // ---- File header (32 bytes) ----
  w.write_u32(nam::namb::MAGIC);
  w.write_u16(nam::namb::FORMAT_VERSION);
  w.write_u16(0); // flags

  const size_t total_size_offset = w.position();
  w.write_u32(0);
  const size_t weights_offset_offset = w.position();
  w.write_u32(0);
  const size_t weight_count_offset = w.position();
  w.write_u32(0);
  const size_t model_block_size_offset = w.position();
  w.write_u32(0);
  w.write_u32(0); // checksum, backpatched at offset 24
  w.write_u32(0); // reserved

  // ---- Metadata block (48 bytes at offset 32) ----
  {
    const Status st = WriteMetadataBlock(ctx, model, outer_meta);
    if (!IsOk(st))
      return st;
  }

  // ---- Model block (variable, from offset 80) ----
  const size_t model_block_start = w.position();
  {
    const Status st = WriteModelBlock(ctx, model);
    if (!IsOk(st))
      return st;
  }
  const size_t model_block_size = w.position() - model_block_start;

  // ---- Weights, 4-byte aligned ----
  w.align_to(4);
  const size_t weights_offset = w.position();

  uint32_t weight_count = 0;
  if (!WeightsInOrder(model, weight_count, &w))
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "'weights' must be an array of numbers");

  // One check for every bounds check skipped above: if the span was too small
  // the writer stopped at the first overrun and everything since is short.
  if (w.failed())
  {
    NAMB_FAIL(ctx, Status::ErrorTooSmall, "output buffer of %u bytes is too small for this model",
              static_cast<unsigned>(capacity));
  }

  // ---- Backpatch ----
  const uint32_t total_size = static_cast<uint32_t>(w.position());
  w.set_u32(total_size_offset, total_size);
  w.set_u32(weights_offset_offset, static_cast<uint32_t>(weights_offset));
  w.set_u32(weight_count_offset, weight_count);
  w.set_u32(model_block_size_offset, static_cast<uint32_t>(model_block_size));
  w.set_u32(24, compute_file_crc32(out, total_size));

  result.size = total_size;
  result.weight_count = weight_count;
  DeriveName(model, outer_meta, opts.fallback_name, result.name, kNameSize);
  return Status::Ok;
}

/// Channel count of a model's first layer array, or -1 if it has none. Used to
/// pick a submodel out of a container.
template<class J>
int LayerArrayChannels(const J& model)
{
  const J* config = Member(model, "config");
  if (config == nullptr)
    return -1;
  const J* layers = Member(*config, "layers");
  if (layers == nullptr || !layers->is_array() || layers->empty())
    return -1;
  const J* channels = Member((*layers)[0], "channels");
  int value = 0;
  if (channels == nullptr || !AsInt(*channels, value))
    return -1;
  return value;
}

} // namespace writer_detail

// =============================================================================
// Entry point
// =============================================================================

/// \brief Convert a parsed .nam document into a .namb blob.
///
/// \param doc      A WaveNet model, or a SlimmableContainer to select from.
/// \param out      Destination span. 16 KiB is comfortable for A2-Lite, whose
///                 blob is a little over 8 KiB.
/// \param capacity Bytes available at \p out.
/// \param result   Size, weight count and suggested name on success; \p detail
///                 carries the reason on failure.
/// \param opts     Which submodel to take out of a container.
/// \return Status::Ok, or why the document could not be converted.
template<class J>
Status WriteNamb(const J& doc, uint8_t* out, size_t capacity, WriteResult& result,
                 const WriteOptions& opts = WriteOptions{})
{
  using namespace writer_detail;

  result = WriteResult{};

  SpanWriter probe(nullptr, 0); // only so a failure before any write has a ctx
  Context<J> ctx{probe, result.detail};

  if (out == nullptr || capacity < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
    NAMB_FAIL(ctx, Status::ErrorTooSmall, "output span is too small to hold even a .namb header");

  const J* arch = Member(doc, "architecture");
  const typename J::string_t* arch_name = (arch != nullptr) ? AsString(*arch) : nullptr;
  if (arch_name == nullptr)
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "document has no string 'architecture'");

  // --- a bare model ---------------------------------------------------------
  if (*arch_name == "WaveNet")
  {
    const Status st = WriteOneModel(doc, static_cast<const J*>(nullptr), out, capacity, result, opts);
    if (IsOk(st))
    {
      const int ch = LayerArrayChannels(doc);
      result.channels = (ch > 0) ? static_cast<uint16_t>(ch) : 0;
    }
    return st;
  }

  // --- a container ----------------------------------------------------------
  if (*arch_name != "SlimmableContainer")
  {
    NAMB_FAIL(ctx, Status::ErrorUnknownArchitecture,
              "architecture '%s' is not supported; expected WaveNet or SlimmableContainer", arch_name->c_str());
  }

  const J* config = Member(doc, "config");
  const J* submodels = (config != nullptr) ? Member(*config, "submodels") : nullptr;
  if (submodels == nullptr || !submodels->is_array() || submodels->empty())
    NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "container has no non-empty 'submodels' array");

  const J* outer_meta = Member(doc, "metadata");

  // Report what was on offer when nothing matches -- "no channels=3 submodel"
  // is not actionable without knowing the file held 8.
  char available[48] = {};
  size_t available_len = 0;

  for (const auto& sm : *submodels)
  {
    const J* model = Member(sm, "model");
    if (model == nullptr || !model->is_object())
      continue;

    const int ch = LayerArrayChannels(*model);
    if (ch > 0 && available_len + 8 < sizeof(available))
    {
      available_len += static_cast<size_t>(
        std::snprintf(available + available_len, sizeof(available) - available_len, available_len ? ", %d" : "%d", ch));
    }

    if (ch < 0 || static_cast<uint16_t>(ch) != opts.channels)
      continue;

    const Status st = WriteOneModel(*model, outer_meta, out, capacity, result, opts);
    if (IsOk(st))
      result.channels = opts.channels;
    return st;
  }

  NAMB_FAIL(ctx, Status::ErrorInvalidConfig, "container has no channels=%u submodel (it has %s)",
            static_cast<unsigned>(opts.channels), available_len ? available : "none");
}

} // namespace namb
} // namespace nam

#undef NAMB_FAIL

