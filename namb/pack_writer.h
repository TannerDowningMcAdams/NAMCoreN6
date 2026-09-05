#pragma once
// Write side of the model pack: add, replace and remove models in place.
//
// A commit programs the blob into free space, then rewrites the INACTIVE TOC
// slot at sequence+1. The live slot is never touched, so a power loss at any
// point leaves the pack exactly as it was.
//
// Flash access goes through FlashOps so the same code runs against the W25Q on
// target and against a simulator on the host. Everything that decides what gets
// erased and what gets programmed lives here, which is the part worth testing
// without a board.

#include <cstddef>
#include <cstdint>
#include <cstring>

#include <NAM/status.h>

#include "nambpack_format.h"
#include "nambpack_reader.h"

namespace nam
{
namespace nambpack
{

/// Flash back end. Addresses are raw chip offsets; \p base is where the pack
/// appears in the memory map.
struct FlashOps
{
  const uint8_t* base = nullptr;
  uint32_t chip_base = 0; ///< Pack offset within the chip
  void* ctx = nullptr;

  bool (*suspend)(void* ctx) = nullptr; ///< Leave memory-mapped mode
  bool (*resume)(void* ctx) = nullptr;
  bool (*erase_sector)(void* ctx, uint32_t chip_sector) = nullptr;

  /// One page program. Never called with len > 256 or across a page boundary.
  bool (*program)(void* ctx, uint32_t chip_addr, const uint8_t* src, uint16_t len) = nullptr;

  void (*invalidate)(void* ctx, const uint8_t* addr, uint32_t len) = nullptr;
};

/// Pack state, for a UI or a "will it fit" decision.
struct PackInfo
{
  uint16_t count = 0;
  uint32_t used_bytes = 0; ///< Blob bytes claimed, sector padding included
  uint32_t largest_free = 0; ///< Biggest contiguous run available
  uint32_t sequence = 0;
  uint8_t active_slot = 0;
};

#pragma pack(push, 1)
/// Header and TOC laid out exactly as they are programmed, so the checksum can
/// be taken over the bytes that actually go down.
struct TocImage
{
  Header header;
  Entry entries[MAX_ENTRIES];
};
#pragma pack(pop)

static_assert(sizeof(TocImage) == sizeof(Header) + MAX_ENTRIES * sizeof(Entry), "TocImage must be contiguous");

class PackWriter
{
public:
  explicit PackWriter(const FlashOps& ops)
  : _ops(ops)
  {
  }

  /// \brief Store \p blob under \p name, replacing any entry of that name.
  ///
  /// \p name is truncated to NAME_SIZE-1. \p blob must NOT be in the flash
  /// being written: it is read with the memory map suspended.
  Status Add(const char* name, const uint8_t* blob, uint32_t size)
  {
    if (name == nullptr || name[0] == '\0' || blob == nullptr || size == 0)
      return Status::ErrorInvalidConfig;

    char key[NAME_SIZE];
    std::memset(key, 0, sizeof(key));
    std::strncpy(key, name, NAME_SIZE - 1);

    bool replaced = false;
    const Status staged = Stage(key, replaced);
    if (!IsOk(staged) && staged != Status::ErrorBadMagic)
      return staged;

    if (_count >= MAX_ENTRIES)
      return Status::ErrorInvalidConfig;

    const uint32_t offset = FindFree(size);
    if (offset == 0)
      return Status::ErrorTooSmall;

    Entry& e = _image.entries[_count++];
    std::memset(&e, 0, sizeof(e));
    e.offset = offset;
    e.size = size;
    std::memcpy(e.name, key, NAME_SIZE);

    FinishHeader();

    const Status st = Commit(offset, blob, size);
    if (!IsOk(st))
      return st;

    return Verify(key, blob, size);
  }

  /// \brief Drop \p name. Its space becomes available to the next Add().
  Status Remove(const char* name)
  {
    if (name == nullptr || name[0] == '\0')
      return Status::ErrorInvalidConfig;

    bool removed = false;
    const Status staged = Stage(name, removed);
    if (!IsOk(staged))
      return staged;
    if (!removed)
      return Status::ErrorInvalidConfig;

    FinishHeader();

    const Status st = Commit(0, nullptr, 0);
    return IsOk(st) ? Verify(nullptr, nullptr, 0) : st;
  }

  /// \brief Write an empty TOC. Blob sectors are left alone, just unreferenced.
  Status Format()
  {
    bool ignored = false;
    const Status staged = Stage(nullptr, ignored);
    if (!IsOk(staged) && staged != Status::ErrorBadMagic)
      return staged;

    _count = 0;
    FinishHeader();

    const Status st = Commit(0, nullptr, 0);
    return IsOk(st) ? Verify(nullptr, nullptr, 0) : st;
  }

  /// \brief Read the pack without writing. Does not suspend the memory map.
  Status Query(PackInfo& info)
  {
    info = PackInfo{};

    bool ignored = false;
    const Status staged = Stage(nullptr, ignored);
    if (!IsOk(staged))
      return staged;

    info.count = _count;
    info.sequence = _sequence;
    info.active_slot = static_cast<uint8_t>((_next_slot + TOC_SLOT_COUNT - 1) % TOC_SLOT_COUNT);

    for (uint16_t i = 0; i < _count; i++)
      info.used_bytes += AlignUp(_image.entries[i].size, SECTOR_SIZE);

    info.largest_free = LargestFree();
    return Status::Ok;
  }

private:
  static constexpr uint32_t kPageSize = 256;

  static constexpr uint32_t AlignUp(uint32_t v, uint32_t a) { return ((v + a - 1u) / a) * a; }

  // --- staging ---------------------------------------------------------------

  /// Copy the live TOC into RAM, dropping \p exclude if present. Must happen
  /// before the map goes down, since it reads flash.
  Status Stage(const char* exclude, bool& excluded)
  {
    excluded = false;
    _count = 0;
    _sequence = 0;
    _next_slot = 0;
    std::memset(&_image, 0, sizeof(_image));

    PackView pack;
    const Status st = pack.Open(_ops.base, REGION_SIZE);
    if (!IsOk(st))
      return st; // ErrorBadMagic here means "no pack yet", handled by callers

    for (uint16_t i = 0; i < pack.Count(); i++)
    {
      const Entry* e = pack.At(i);
      if (exclude != nullptr && std::strncmp(e->name, exclude, NAME_SIZE) == 0)
      {
        excluded = true;
        continue;
      }
      _image.entries[_count++] = *e;
    }

    _sequence = pack.Sequence();
    _next_slot = pack.NextSlot();
    return Status::Ok;
  }

  /// Occupied runs, sector-aligned outward and sorted by start. Aligning
  /// outward is what keeps a new blob from ever sharing an erase sector with an
  /// existing one, whatever alignment the pack was built with.
  uint16_t SortedRuns(uint32_t* start, uint32_t* end) const
  {
    for (uint16_t i = 0; i < _count; i++)
    {
      const uint32_t s = (_image.entries[i].offset / SECTOR_SIZE) * SECTOR_SIZE;
      const uint32_t e = AlignUp(_image.entries[i].offset + _image.entries[i].size, SECTOR_SIZE);

      uint16_t j = i;
      while (j > 0 && start[j - 1] > s)
      {
        start[j] = start[j - 1];
        end[j] = end[j - 1];
        j--;
      }
      start[j] = s;
      end[j] = e;
    }
    return _count;
  }

  /// First-fit. Returns 0 when there is no room; a blob can never sit at 0.
  uint32_t FindFree(uint32_t bytes) const
  {
    const uint32_t need = AlignUp(bytes, SECTOR_SIZE);
    uint32_t start[MAX_ENTRIES];
    uint32_t end[MAX_ENTRIES];
    const uint16_t n = SortedRuns(start, end);

    uint32_t cursor = FIRST_BLOB_OFFSET;
    for (uint16_t i = 0; i < n; i++)
    {
      if (start[i] >= cursor + need)
        return cursor;
      if (end[i] > cursor)
        cursor = end[i];
    }
    return (REGION_SIZE - cursor >= need) ? cursor : 0;
  }

  uint32_t LargestFree() const
  {
    uint32_t start[MAX_ENTRIES];
    uint32_t end[MAX_ENTRIES];
    const uint16_t n = SortedRuns(start, end);

    uint32_t best = 0;
    uint32_t cursor = FIRST_BLOB_OFFSET;
    for (uint16_t i = 0; i < n; i++)
    {
      if (start[i] > cursor && start[i] - cursor > best)
        best = start[i] - cursor;
      if (end[i] > cursor)
        cursor = end[i];
    }
    if (REGION_SIZE - cursor > best)
      best = REGION_SIZE - cursor;
    return best;
  }

  void FinishHeader()
  {
    uint32_t total = FIRST_BLOB_OFFSET;
    for (uint16_t i = 0; i < _count; i++)
    {
      const uint32_t end = _image.entries[i].offset + _image.entries[i].size;
      if (end > total)
        total = end;
    }

    Header& h = _image.header;
    std::memset(&h, 0, sizeof(h));
    h.magic = MAGIC;
    h.version = FORMAT_VERSION;
    h.count = _count;
    h.total_size = total;
    h.toc_crc32 = TocCrc32(reinterpret_cast<const uint8_t*>(&_image), _count);
    h.sequence = _sequence + 1;
    h.header_crc32 = HeaderCrc32(h);
  }

  // --- flash -----------------------------------------------------------------

  bool EraseRange(uint32_t offset, uint32_t len)
  {
    if (len == 0)
      return true;
    const uint32_t first = (_ops.chip_base + offset) / SECTOR_SIZE;
    const uint32_t last = (_ops.chip_base + offset + len - 1) / SECTOR_SIZE;
    for (uint32_t s = first; s <= last; s++)
    {
      if (!_ops.erase_sector(_ops.ctx, s))
        return false;
    }
    return true;
  }

  /// A NOR page program wraps within its 256-byte page rather than crossing
  /// into the next, so the split is on page boundaries, not on 256 bytes.
  bool ProgramRange(uint32_t offset, const uint8_t* src, uint32_t len)
  {
    uint32_t addr = _ops.chip_base + offset;
    while (len != 0)
    {
      uint32_t n = kPageSize - (addr % kPageSize);
      if (n > len)
        n = len;
      if (!_ops.program(_ops.ctx, addr, src, static_cast<uint16_t>(n)))
        return false;
      addr += n;
      src += n;
      len -= n;
    }
    return true;
  }

  /// The write window. \p blob_offset of 0 means TOC only.
  Status Commit(uint32_t blob_offset, const uint8_t* blob, uint32_t size)
  {
    if (!_ops.suspend(_ops.ctx))
      return Status::Error;

    Status result = Status::Ok;

    if (blob_offset != 0)
    {
      if (!EraseRange(blob_offset, size) || !ProgramRange(blob_offset, blob, size))
        result = Status::Error;
    }

    // Only once the blob is down: the TOC is what makes it reachable, so
    // failing before this leaves the pack referencing nothing new.
    if (IsOk(result))
    {
      const uint32_t toc = TocSlotOffset(_next_slot);
      const uint32_t live = sizeof(Header) + _count * sizeof(Entry);
      if (!EraseRange(toc, TOC_SLOT_SIZE) || !ProgramRange(toc, reinterpret_cast<const uint8_t*>(&_image), live))
        result = Status::Error;
    }

    // Unconditional: leaving the map down because a write failed turns a lost
    // model into an image that cannot read its own LUTs.
    if (!_ops.resume(_ops.ctx))
      result = Status::Error;

    if (_ops.invalidate != nullptr)
    {
      if (blob_offset != 0)
        _ops.invalidate(_ops.ctx, _ops.base + blob_offset, size);
      _ops.invalidate(_ops.ctx, _ops.base + TocSlotOffset(_next_slot), TOC_SLOT_SIZE);
    }

    return result;
  }

  /// Re-read through the restored map: checks the bytes landed AND that the
  /// mapping came back, in one go.
  Status Verify(const char* name, const uint8_t* blob, uint32_t size)
  {
    PackView pack;
    const Status st = pack.Open(_ops.base, REGION_SIZE);
    if (!IsOk(st))
      return st;
    if (pack.ActiveSlot() != _next_slot)
      return Status::ErrorChecksum;

    if (name == nullptr)
      return Status::Ok;

    const Entry* e = pack.Find(name);
    if (e == nullptr || e->size != size)
      return Status::ErrorChecksum;
    return (std::memcmp(pack.Blob(*e), blob, size) == 0) ? Status::Ok : Status::ErrorChecksum;
  }

  FlashOps _ops;
  TocImage _image{};
  uint16_t _count = 0;
  uint32_t _sequence = 0;
  uint8_t _next_slot = 0;
};

} // namespace nambpack
} // namespace nam
