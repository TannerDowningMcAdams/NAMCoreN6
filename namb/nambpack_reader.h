#pragma once
// Read-side view of a model pack sitting in memory-mapped flash. Turns the raw
// bytes at FLASH_BASE into validated Entry records and hands back pointers into
// that same window; nothing is copied out, since get_dsp_namb() parses through
// a pointer and a length. Header-only and allocation-free, so it works before
// the heap is warm.
//
//   nam::nambpack::PackView pack;
//   if (nam::IsOk(pack.Open()))
//   {
//       const auto* e = pack.Find("Deluxe_Reverb_sub0_ch3");   // or pack.At(0)
//       nam::Status st;
//       auto model = nam::get_dsp_namb(pack.Blob(*e), e->size, st);
//   }
//
// Flash contents may be erased (all 0xFF), stale from an older layout, or
// half-programmed, so every field is range checked before use: a bad offset
// would otherwise be dereferenced straight into a fault.

#include <cstdint>
#include <cstring>

#include "namb_format.h" // crc32
#include "nambpack_format.h"

#include <NAM/status.h>

namespace nam
{
namespace nambpack
{

/// \brief Validated, non-owning view of a pack image.
///
/// Entry records are read in place rather than copied: Header is 32 bytes and
/// Entry is 40, both multiples of 4, so every record is naturally aligned for
/// its uint32_t fields and can be addressed directly in the flash window.
class PackView
{
public:
  PackView() = default;

  /// \brief Validate the pack at \p base and make its entries available.
  ///
  /// \param base        Start of the pack. Defaults to the memory-mapped
  ///                    location; pass something else to validate a copy in RAM.
  /// \param region_size Bytes the pack is allowed to occupy. Anything claiming
  ///                    more is rejected rather than trusted, since past this
  ///                    point is user data.
  /// \return Status::Ok, or the reason the image was rejected.
  Status Open(const uint8_t* base = reinterpret_cast<const uint8_t*>(FLASH_BASE),
              uint32_t region_size = REGION_SIZE)
  {
    _base = nullptr;
    _count = 0;

    if (base == nullptr || region_size < sizeof(Header))
      return Status::ErrorTooSmall;

    const Header* h = reinterpret_cast<const Header*>(base);

    // Erased flash reads 0xFFFFFFFF, so this is also the "no pack programmed"
    // case -- not an error worth distinguishing from a corrupt one here.
    if (h->magic != MAGIC)
      return Status::ErrorBadMagic;

    if (h->version != FORMAT_VERSION)
      return Status::ErrorUnsupportedVersion;

    if (h->count > MAX_ENTRIES)
      return Status::ErrorInvalidConfig;

    if (h->total_size > region_size)
      return Status::ErrorWeightsOutOfRange;

    // TOC must fit inside what the header claims, before it is hashed or read.
    const uint32_t toc_bytes = static_cast<uint32_t>(h->count) * sizeof(Entry);
    if (static_cast<uint64_t>(sizeof(Header)) + toc_bytes > h->total_size)
      return Status::ErrorTruncated;

    if (nam::namb::crc32(base + sizeof(Header), toc_bytes) != h->toc_crc32)
      return Status::ErrorChecksum;

    const Entry* entries = reinterpret_cast<const Entry*>(base + sizeof(Header));

    // Every blob must lie inside the image and clear of the TOC. A stale or
    // hand-edited pack is the realistic way a bad offset gets here, and the
    // cost of trusting one is a fault at model-load time.
    for (uint16_t i = 0; i < h->count; i++)
    {
      const Entry& e = entries[i];
      if (e.size == 0)
        return Status::ErrorInvalidConfig;
      if (e.offset < sizeof(Header) + toc_bytes)
        return Status::ErrorInvalidConfig;
      if (static_cast<uint64_t>(e.offset) + e.size > h->total_size)
        return Status::ErrorWeightsOutOfRange;
      if (e.name[NAME_SIZE - 1] != '\0') // name must be NUL-terminated
        return Status::ErrorInvalidConfig;
    }

    _base = base;
    _entries = entries;
    _count = h->count;
    _total_size = h->total_size;
    return Status::Ok;
  }

  /// \brief True once Open() has succeeded.
  bool IsOpen() const { return _base != nullptr; }

  /// \brief Number of models in the pack.
  uint16_t Count() const { return _count; }

  /// \brief Bytes the pack image occupies.
  uint32_t TotalSize() const { return _total_size; }

  /// \brief Entry by index, or nullptr if out of range.
  const Entry* At(uint16_t index) const
  {
    if (!IsOpen() || index >= _count)
      return nullptr;
    return &_entries[index];
  }

  /// \brief Entry by name, or nullptr if absent. Names are unique -- nambpack
  ///        rejects duplicates at pack time.
  const Entry* Find(const char* name) const
  {
    if (!IsOpen() || name == nullptr)
      return nullptr;
    for (uint16_t i = 0; i < _count; i++)
    {
      if (std::strncmp(_entries[i].name, name, NAME_SIZE) == 0)
        return &_entries[i];
    }
    return nullptr;
  }

  /// \brief Pointer to an entry's .namb blob, inside the flash window.
  ///        Valid to hand straight to get_dsp_namb(); no copy required.
  const uint8_t* Blob(const Entry& e) const { return _base + e.offset; }

private:
  const uint8_t* _base = nullptr;
  const Entry* _entries = nullptr;
  uint16_t _count = 0;
  uint32_t _total_size = 0;
};

} // namespace nambpack
} // namespace nam
