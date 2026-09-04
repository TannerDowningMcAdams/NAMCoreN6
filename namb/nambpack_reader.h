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
//
// A v2 pack carries two TOC slots (see nambpack_format.h). Open() validates
// both independently and takes the one with the higher sequence, so a commit
// interrupted by a power loss falls back to the pack as it was beforehand
// rather than to nothing at all. ActiveSlot() reports which one won, which is
// what a writer needs in order to target the other.

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
/// Entry is 40, both multiples of 4, and a TOC slot starts on an erase-sector
/// boundary, so every record is naturally aligned for its uint32_t fields and
/// can be addressed directly in the flash window.
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
  /// \return Status::Ok, or the reason both TOC slots were rejected.
  Status Open(const uint8_t* base = reinterpret_cast<const uint8_t*>(FLASH_BASE),
              uint32_t region_size = REGION_SIZE)
  {
    _base = nullptr;
    _entries = nullptr;
    _count = 0;
    _total_size = 0;
    _sequence = 0;
    _slot = 0;

    if (base == nullptr || region_size < FIRST_BLOB_OFFSET)
      return Status::ErrorTooSmall;

    // Validate every slot before choosing, so a torn write to one cannot stop
    // the other from being considered.
    Status slot_status[TOC_SLOT_COUNT];
    uint32_t slot_sequence[TOC_SLOT_COUNT];

    for (uint8_t s = 0; s < TOC_SLOT_COUNT; s++)
      slot_status[s] = ValidateSlot(base, region_size, s, slot_sequence[s]);

    // Highest sequence wins. Ties go to the lower slot index rather than to
    // whichever the loop saw last, so a pack whose two slots somehow agree
    // resolves identically on every boot.
    uint8_t winner = TOC_SLOT_COUNT; // sentinel: none valid
    for (uint8_t s = 0; s < TOC_SLOT_COUNT; s++)
    {
      if (!IsOk(slot_status[s]))
        continue;
      if (winner == TOC_SLOT_COUNT || SequenceNewer(slot_sequence[s], slot_sequence[winner]))
        winner = s;
    }

    if (winner == TOC_SLOT_COUNT)
    {
      // Nothing usable. Report the most informative failure: erased flash reads
      // as ErrorBadMagic on every slot and means "no pack programmed", which is
      // a different situation from a pack that is present and damaged.
      for (uint8_t s = 0; s < TOC_SLOT_COUNT; s++)
      {
        if (slot_status[s] != Status::ErrorBadMagic)
          return slot_status[s];
      }
      return Status::ErrorBadMagic;
    }

    const uint8_t* slot = base + TocSlotOffset(winner);
    const Header* h = reinterpret_cast<const Header*>(slot);

    _base = base;
    _entries = reinterpret_cast<const Entry*>(slot + sizeof(Header));
    _count = h->count;
    _total_size = h->total_size;
    _sequence = h->sequence;
    _slot = winner;
    return Status::Ok;
  }

  /// \brief True once Open() has succeeded.
  bool IsOpen() const { return _base != nullptr; }

  /// \brief Number of models in the pack. Zero is valid: a pack that has been
  ///        initialised but holds no models yet.
  uint16_t Count() const { return _count; }

  /// \brief Bytes the pack image occupies, TOC area included.
  uint32_t TotalSize() const { return _total_size; }

  /// \brief Commit counter of the TOC in force. A writer stores this plus one.
  uint32_t Sequence() const { return _sequence; }

  /// \brief Which TOC slot won arbitration. A writer commits to the other one,
  ///        so the live index survives a power loss mid-write.
  uint8_t ActiveSlot() const { return _slot; }

  /// \brief The slot a writer should commit its next TOC to.
  uint8_t NextSlot() const { return static_cast<uint8_t>((_slot + 1u) % TOC_SLOT_COUNT); }

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

  /// \brief Validate one TOC slot without disturbing the open view.
  ///
  /// Exposed because the packing tool reports on both slots, and because a
  /// writer wants to know whether the slot it is about to overwrite was the
  /// one carrying the live index.
  ///
  /// \param sequence_out Set to the slot's commit counter on success.
  static Status ValidateSlot(const uint8_t* base, uint32_t region_size, uint8_t slot, uint32_t& sequence_out)
  {
    sequence_out = 0;

    if (base == nullptr || slot >= TOC_SLOT_COUNT)
      return Status::ErrorInvalidConfig;

    const uint32_t slot_offset = TocSlotOffset(slot);
    if (static_cast<uint64_t>(slot_offset) + TOC_SLOT_SIZE > region_size)
      return Status::ErrorTooSmall;

    const uint8_t* p = base + slot_offset;
    const Header* h = reinterpret_cast<const Header*>(p);

    // Erased flash reads 0xFFFFFFFF, so this is also the "no pack programmed"
    // case -- not an error worth distinguishing from a corrupt one here.
    if (h->magic != MAGIC)
      return Status::ErrorBadMagic;

    if (h->version != FORMAT_VERSION)
      return Status::ErrorUnsupportedVersion;

    // Before anything else in the header is believed, including the sequence
    // that decides this slot against its neighbour.
    if (HeaderCrc32(*h) != h->header_crc32)
      return Status::ErrorChecksum;

    if (h->count > MAX_ENTRIES)
      return Status::ErrorInvalidConfig;

    if (h->total_size > region_size)
      return Status::ErrorWeightsOutOfRange;

    // Blobs start after the whole TOC area whatever the count, so an image that
    // does not even reach that point cannot hold one.
    if (h->total_size < FIRST_BLOB_OFFSET)
      return Status::ErrorTruncated;

    if (TocCrc32(p, h->count) != h->toc_crc32)
      return Status::ErrorChecksum;

    const Entry* entries = reinterpret_cast<const Entry*>(p + sizeof(Header));

    // Every blob must lie inside the image and clear of the TOC area. A stale
    // or hand-edited pack is the realistic way a bad offset gets here, and the
    // cost of trusting one is a fault at model-load time.
    for (uint16_t i = 0; i < h->count; i++)
    {
      const Entry& e = entries[i];
      if (e.size == 0)
        return Status::ErrorInvalidConfig;
      if (e.offset < FIRST_BLOB_OFFSET)
        return Status::ErrorInvalidConfig;
      if (static_cast<uint64_t>(e.offset) + e.size > h->total_size)
        return Status::ErrorWeightsOutOfRange;
      if (e.name[NAME_SIZE - 1] != '\0') // name must be NUL-terminated
        return Status::ErrorInvalidConfig;
    }

    sequence_out = h->sequence;
    return Status::Ok;
  }

private:
  const uint8_t* _base = nullptr;
  const Entry* _entries = nullptr;
  uint16_t _count = 0;
  uint32_t _total_size = 0;
  uint32_t _sequence = 0;
  uint8_t _slot = 0;
};

} // namespace nambpack
} // namespace nam
