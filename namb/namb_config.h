#pragma once
// The seam between "read a model description" and "lay out .namb bytes".
//
// Everything here is plain structs and the code that turns them into bytes.
// Nothing in this file knows what JSON is. That is the entire point: the byte
// layout is the part that has been verified against goldens produced by the
// pipeline that built every pack ever flashed to a pedal, and it must not have
// to change again when the front end that fills these structs does.
//
//   source --[front end]--> LayerArrayCfg &c. --[Emit*]--> .namb bytes
//                                ^                              ^
//                        namb_writer.h today,          this file, byte-verified
//                        a streaming parser next
//
// The structs are bounded and contain no pointers, so a front end can fill one
// field at a time as a document streams past, without holding the document.
//
// SIZE. LayerArrayCfg is ~12 KB and is deliberately not something to put on the
// stack -- the target's MSP stack is 8 KiB. Callers own a ModelScratch and pass
// it in; ModelLibrary keeps one as a member alongside its output blob.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "namb_format.h"

namespace nam
{
namespace namb
{

/// \brief Largest dilation count a layer array may declare. Not an arbitrary
///        cap: num_dilations is a uint8 in the container, so 255 is the
///        format's own limit.
static constexpr size_t kMaxDilations = 255;

/// \brief FiLM blocks per layer array. Fixed by the format.
static constexpr size_t kNumFilmBlocks = 8;

/// \brief Floats available to one layer array's activation parameters, shared
///        between the primary and secondary blocks.
///
/// Sized so that every activation type except PReLU-per-channel can use its
/// maximum on all 255 dilations, twice over: LeakyHardtanh is the greediest at
/// 4 parameters, and 255 x 4 x 2 = 2040.
///
/// PReLU with a "negative_slopes" array is the one thing this bounds more
/// tightly than the format does. The format allows 255 slopes per activation,
/// which at 255 dilations would be 65025 floats -- an amount no bounded struct
/// can hold, and the reason the DOM had to exist. A model that exceeds the pool
/// is refused with a sentence saying so rather than silently truncated. No
/// shipping model uses PReLU at all.
static constexpr size_t kActivationParamPool = 2048;

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

  /// \brief Backpatch a uint16. set_u32 would clobber the two bytes after the
  ///        field, so the config-size patch needs its own width.
  void set_u16(size_t offset, uint16_t v)
  {
    if (_failed || offset + 2 > _pos)
      return;
    std::memcpy(_data + offset, &v, 2);
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

// =============================================================================
// Configuration structs
// =============================================================================

/// \brief One activation config, as .namb encodes it: a type id and a
///        parameter list.
///
/// The parameters live in the owning LayerArrayCfg's pool rather than inline,
/// which keeps this at 4 bytes and lets a PReLU's per-channel slopes use the
/// same representation as a LeakyReLU's single slope.
struct ActivationCfg
{
  uint8_t type = 0; ///< nam::activations::ActivationType ordinal
  uint8_t param_count = 0; ///< Floats at \c param_offset in the pool
  uint16_t param_offset = 0; ///< Index into LayerArrayCfg::params
};

/// \brief One FiLM block (4 bytes on the wire).
///
/// An absent block and a block whose "active" is false are not the same bytes,
/// so the front end resolves both into \c flags rather than a "present" bit --
/// absent is flags 0, {"active":false,"shift":true} is flags 0x02, and the
/// loader reads shift independently of active.
struct FilmCfg
{
  uint8_t flags = 0;
  uint16_t groups = 1;
};

/// \brief One WaveNet layer array, complete.
///
/// ~12 KB. Not for the stack; see the file header.
struct LayerArrayCfg
{
  uint16_t input_size = 0;
  uint16_t condition_size = 0;
  uint16_t head_size = 0;
  uint16_t channels = 0;
  uint16_t bottleneck = 0;
  uint16_t head_kernel_size = 1; ///< v1 held a shared kernel_size in this field
  uint8_t head_bias = 0;
  uint8_t num_dilations = 0;
  uint16_t groups_input = 1;
  uint16_t groups_input_mixin = 1;
  int32_t head_dilation = 1; ///< v2

  uint8_t layer1x1_active = 1;
  uint16_t layer1x1_groups = 1;

  uint8_t head1x1_active = 0;
  uint16_t head1x1_out_channels = 0;
  uint16_t head1x1_groups = 1;

  FilmCfg film[kNumFilmBlocks];

  int32_t dilations[kMaxDilations] = {};
  int32_t kernel_sizes[kMaxDilations] = {}; ///< v2; a scalar kernel_size is broadcast here

  ActivationCfg activations[kMaxDilations];
  uint8_t gating[kMaxDilations] = {};

  /// One per dilation whatever the gating, so the block is a fixed shape; the
  /// loader only reads the ones whose gating mode is not NONE. The front end
  /// fills the unused slots with the placeholder the format expects.
  ActivationCfg secondary[kMaxDilations];

  float params[kActivationParamPool] = {};
  uint16_t param_used = 0; ///< Floats claimed from \c params so far

  /// \brief Claim \p n floats from the pool.
  /// \return Pointer to write them through, or nullptr if the pool is full.
  float* claim(uint8_t n, uint16_t& offset)
  {
    if (static_cast<size_t>(param_used) + n > kActivationParamPool)
      return nullptr;
    offset = param_used;
    float* p = params + param_used;
    param_used = static_cast<uint16_t>(param_used + n);
    return p;
  }

  /// \brief Start a fresh layer array. Only the pool cursor and the counts
  ///        carry state between arrays, so this is cheaper than reassigning
  ///        12 KB of struct.
  void reset()
  {
    param_used = 0;
    num_dilations = 0;
  }
};

/// \brief The 48-byte metadata block.
struct MetadataCfg
{
  uint8_t version[3] = {0, 0, 0};
  uint8_t flags = 0; ///< META_HAS_*
  double sample_rate = -1.0;
  double loudness = 0.0;
  double input_level = 0.0;
  double output_level = 0.0;
};

/// \brief The 4-byte WaveNet config header that precedes condition_dsp and the
///        layer arrays.
struct WaveNetCfg
{
  uint8_t in_channels = 1;
  uint8_t has_head = 0;
  uint8_t num_layer_arrays = 0;
  uint8_t has_condition_dsp = 0;
};

/// \brief Working storage a conversion needs but an output span cannot hold.
///
/// One layer array at a time is enough: the emit order is per-array, and a
/// streaming front end sees them in document order too. Callers own this so it
/// is neither a stack local nor a static.
struct ModelScratch
{
  LayerArrayCfg layer;
};

// =============================================================================
// Emitters
// =============================================================================
//
// Byte-for-byte the layout the goldens pin. None of these can fail: the span
// is checked once at the end, and every value has already been range-checked
// by the front end that filled the struct.

/// \brief Offsets of the file-header fields that are backpatched once the
///        sizes are known.
struct FileHeaderPatch
{
  size_t total_size = 0;
  size_t weights_offset = 0;
  size_t weight_count = 0;
  size_t model_block_size = 0;
};

/// \brief Emit the 32-byte file header with its size fields zeroed, recording
///        where they went.
inline void EmitFileHeader(SpanWriter& w, FileHeaderPatch& patch)
{
  w.write_u32(MAGIC);
  w.write_u16(FORMAT_VERSION);
  w.write_u16(0); // flags

  patch.total_size = w.position();
  w.write_u32(0);
  patch.weights_offset = w.position();
  w.write_u32(0);
  patch.weight_count = w.position();
  w.write_u32(0);
  patch.model_block_size = w.position();
  w.write_u32(0);

  w.write_u32(0); // checksum, backpatched at offset 24
  w.write_u32(0); // reserved
}

/// \brief Fill in the header's size fields and the CRC. The CRC covers the
///        finished file, so it has to be last.
inline void FinalizeFileHeader(SpanWriter& w, const FileHeaderPatch& patch, uint32_t total_size,
                               uint32_t weights_offset, uint32_t weight_count, uint32_t model_block_size,
                               const uint8_t* out)
{
  w.set_u32(patch.total_size, total_size);
  w.set_u32(patch.weights_offset, weights_offset);
  w.set_u32(patch.weight_count, weight_count);
  w.set_u32(patch.model_block_size, model_block_size);
  w.set_u32(24, compute_file_crc32(out, total_size));
}

/// \brief Emit the 48-byte metadata block.
inline void EmitMetadataBlock(SpanWriter& w, const MetadataCfg& m)
{
  w.write_u8(m.version[0]);
  w.write_u8(m.version[1]);
  w.write_u8(m.version[2]);
  w.write_u8(m.flags);
  w.write_f64(m.sample_rate);
  w.write_f64(m.loudness);
  w.write_f64(m.input_level);
  w.write_f64(m.output_level);
  w.write_zeros(12); // reserved
}

/// \brief Emit one activation: type byte, parameter count, then the floats.
inline void EmitActivation(SpanWriter& w, const ActivationCfg& a, const float* pool)
{
  w.write_u8(a.type);
  w.write_u8(a.param_count);
  for (uint8_t i = 0; i < a.param_count; i++)
    w.write_f32(pool[a.param_offset + i]);
}

/// \brief Emit one FiLM block (4 bytes).
inline void EmitFilm(SpanWriter& w, const FilmCfg& f)
{
  w.write_u8(f.flags);
  w.write_u8(0); // reserved
  w.write_u16(f.groups);
}

/// \brief Emit the 4-byte WaveNet config header.
inline void EmitWaveNetConfigHeader(SpanWriter& w, const WaveNetCfg& c)
{
  w.write_u8(c.in_channels);
  w.write_u8(c.has_head);
  w.write_u8(c.num_layer_arrays);
  w.write_u8(c.has_condition_dsp);
}

/// \brief Emit one complete layer array.
inline void EmitLayerArray(SpanWriter& w, const LayerArrayCfg& la)
{
  const size_t n = la.num_dilations;

  // --- fixed block ----------------------------------------------------------
  w.write_u16(la.input_size);
  w.write_u16(la.condition_size);
  w.write_u16(la.head_size);
  w.write_u16(la.channels);
  w.write_u16(la.bottleneck);
  w.write_u16(la.head_kernel_size);
  w.write_u8(la.head_bias);
  w.write_u8(la.num_dilations);
  w.write_u16(la.groups_input);
  w.write_u16(la.groups_input_mixin);
  w.write_i32(la.head_dilation);

  // --- layer1x1 (4 bytes) ---------------------------------------------------
  w.write_u8(la.layer1x1_active);
  w.write_u16(la.layer1x1_groups);
  w.write_u8(0); // reserved

  // --- head1x1 (6 bytes) ----------------------------------------------------
  w.write_u8(la.head1x1_active);
  w.write_u16(la.head1x1_out_channels);
  w.write_u16(la.head1x1_groups);
  w.write_u8(0); // reserved

  // --- 8 FiLM blocks (32 bytes) ---------------------------------------------
  for (size_t i = 0; i < kNumFilmBlocks; i++)
    EmitFilm(w, la.film[i]);

  // --- dilations, then kernel sizes -----------------------------------------
  for (size_t i = 0; i < n; i++)
    w.write_i32(la.dilations[i]);
  for (size_t i = 0; i < n; i++)
    w.write_i32(la.kernel_sizes[i]); // v2

  // --- activations, gating modes, secondary activations ---------------------
  for (size_t i = 0; i < n; i++)
    EmitActivation(w, la.activations[i], la.params);
  for (size_t i = 0; i < n; i++)
    w.write_u8(la.gating[i]);
  for (size_t i = 0; i < n; i++)
    EmitActivation(w, la.secondary[i], la.params);
}

/// \brief Offsets a model block needs in order to backpatch its config size.
struct ModelBlockPatch
{
  size_t config_size_offset = 0;
  size_t config_start = 0;
};

/// \brief Emit a model block's header, leaving the config size to be filled in.
inline ModelBlockPatch EmitModelBlockOpen(SpanWriter& w, uint8_t arch)
{
  w.write_u8(arch);
  w.write_u8(0); // reserved

  ModelBlockPatch patch;
  patch.config_size_offset = w.position();
  w.write_u16(0); // backpatched by EmitModelBlockClose
  patch.config_start = w.position();
  return patch;
}

/// \brief Close a model block, patching in the size of the config that was
///        written since it opened.
/// \return false if the config is too large for its uint16 length; \p size is
///         set either way so the caller can say how large it was.
inline bool EmitModelBlockClose(SpanWriter& w, const ModelBlockPatch& patch, size_t& size)
{
  size = w.position() - patch.config_start;
  if (size > 65535)
    return false;
  w.set_u16(patch.config_size_offset, static_cast<uint16_t>(size));
  return true;
}

} // namespace namb
} // namespace nam
