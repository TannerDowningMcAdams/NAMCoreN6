#pragma once
// Model pack (.nambpack): two redundant tables of contents plus N .namb blobs,
// programmed to external flash as one image and read in place.
//
// Shared by the packing tool and the firmware so the two cannot drift. The
// firmware never copies a model out of flash: the whole pack is read through
// the memory-mapped XSPI window.
//
// Flash map (STM32N6 / W25Q, XSPI1 memory-mapped at 0x90000000):
//
//   0x90200000  +--------------------------+
//               |  model pack   (6 MiB)    |  <- this format
//   0x90800000  +--------------------------+
//               |  user data   (24 MiB)    |  <- logs, recordings, LUTs
//   0x92000000  +--------------------------+
//
// Pack layout, version 2:
//
//   +0x0000  +--------------------------+
//            |  TOC slot A     (8 KiB)  |  Header + MAX_ENTRIES Entry records,
//   +0x2000  +--------------------------+  padded out to two erase sectors
//            |  TOC slot B     (8 KiB)  |
//   +0x4000  +--------------------------+  <- FIRST_BLOB_OFFSET
//            |  blob 0                  |  each sector-aligned
//            |  blob 1                  |
//            |  ...                     |
//            +--------------------------+  <- Header::total_size
//
// Two properties of this layout exist for the firmware's benefit and cost the
// host tool nothing:
//
//   The TOC area is a FIXED size, sized for MAX_ENTRIES rather than for the
//   entries actually present. v1 put the first blob immediately after `count`
//   entries, so adding a model moved every blob in the pack. That is fine for a
//   tool that rewrites the whole image and impossible for a target that grows a
//   pack in place, which is the entire point of this version.
//
//   There are TWO TOC slots, arbitrated by `sequence`. Committing a model means
//   erasing and reprogramming a TOC, and a power loss inside that window would
//   take the index for every model in the pack with it. Alternating slots leaves
//   the previous TOC intact throughout: it describes the pack as it was before
//   the new model, which is exactly the right thing to fall back to. Same idiom
//   as the sense_cal A/B calibration slots.
//
// A v1 image is rejected outright rather than read under its old layout. The
// two disagree about where blob 0 begins, so there is no reading a v1 pack
// "carefully" -- reflash with the current nambpack.
//
// Everything is little-endian, matching the .namb container and the target.

#include <cstddef>
#include <cstdint>

// For crc32(). The checksums below are part of the format contract, so they
// live here with the structures they cover rather than being reimplemented on
// each side. Costs this header namb_format.h's <string>/<vector>, which every
// consumer of this file already pulls in by other routes.
#include "namb_format.h"

namespace nam
{
namespace nambpack
{

/// \brief Magic number: "NMPK" as a little-endian uint32.
static constexpr uint32_t MAGIC = 0x4B504D4Eu;

/// \brief Layout version. Bump on any incompatible change to the structures
///        below. v2 fixed the TOC area at MAX_ENTRIES and added the A/B slots.
static constexpr uint16_t FORMAT_VERSION = 2;

/// \brief Base address of the pack in the memory-mapped XSPI1 window.
static constexpr uint32_t FLASH_BASE = 0x90200000u;

/// \brief Bytes reserved for the pack. The region after this belongs to user
///        data; a pack that would exceed it must be rejected by the tool rather
///        than silently overwriting logs and recordings.
static constexpr uint32_t REGION_SIZE = 6u * 1024u * 1024u;

/// \brief Erase granularity of the W25Q. Blobs are aligned to this so a single
///        model can be replaced with a sector erase instead of rewriting the
///        whole pack -- worth the padding on a rig where models are iterated on.
static constexpr uint32_t SECTOR_SIZE = 4096u;

/// \brief Upper bound on entries, and the size the TOC area is cut to whatever
///        the live count. Keeps the TOC a fixed, bounded read for the firmware
///        and keeps blob offsets stable as models are added.
static constexpr uint16_t MAX_ENTRIES = 128;

/// \brief Characters reserved for an entry name, including the NUL.
static constexpr size_t NAME_SIZE = 32;

#pragma pack(push, 1)

/// \brief Fixed 32-byte header at the start of each TOC slot.
struct Header
{
  uint32_t magic; ///< MAGIC
  uint16_t version; ///< FORMAT_VERSION
  uint16_t count; ///< Number of valid Entry records that follow (0..MAX_ENTRIES)
  uint32_t total_size; ///< Bytes of the whole pack image, TOC area included
  uint32_t toc_crc32; ///< CRC32 over count * sizeof(Entry) bytes of TOC
  uint32_t sequence; ///< Advances by one per commit; the higher one wins
  uint32_t header_crc32; ///< CRC32 over this header, this field excluded
  uint32_t reserved[2]; ///< Zero
};

/// \brief One model. \p offset is relative to the start of the pack, so the
///        image is position-independent and the firmware adds FLASH_BASE.
struct Entry
{
  uint32_t offset; ///< Byte offset of the .namb blob from the pack base
  uint32_t size; ///< Blob length in bytes
  char name[NAME_SIZE]; ///< NUL-terminated, NUL-padded
};

#pragma pack(pop)

static_assert(sizeof(Header) == 32, "nambpack::Header must be 32 bytes");
static_assert(sizeof(Entry) == 40, "nambpack::Entry must be 40 bytes");

/// \brief Byte offset of Header::header_crc32 within Header, and its width.
///        Named because both the writer and the reader have to skip exactly
///        these bytes when hashing, and an off-by-four here is a checksum that
///        never matches on one side only.
static constexpr size_t HEADER_CRC_OFFSET = 20;
static constexpr size_t HEADER_CRC_SIZE = 4;

/// \brief Number of TOC slots. Two is what makes a commit power-safe; there is
///        no reason to have more, and the reader's arbitration assumes this.
static constexpr uint8_t TOC_SLOT_COUNT = 2;

/// \brief Bytes each TOC slot occupies: the header plus a full MAX_ENTRIES
///        table, rounded up to whole erase sectors so a slot can be rewritten
///        without touching its neighbour.
static constexpr uint32_t TOC_SLOT_SIZE =
  ((static_cast<uint32_t>(sizeof(Header)) + MAX_ENTRIES * static_cast<uint32_t>(sizeof(Entry)) + SECTOR_SIZE - 1u)
   / SECTOR_SIZE)
  * SECTOR_SIZE;

/// \brief Byte offset at which blob storage begins. A CONSTANT, deliberately:
///        v1 derived this from the entry count, which is what made growing a
///        pack in place impossible.
static constexpr uint32_t FIRST_BLOB_OFFSET = TOC_SLOT_COUNT * TOC_SLOT_SIZE;

/// \brief Byte offset of TOC slot \p slot from the pack base.
inline constexpr uint32_t TocSlotOffset(uint8_t slot)
{
  return static_cast<uint32_t>(slot) * TOC_SLOT_SIZE;
}

static_assert(REGION_SIZE % SECTOR_SIZE == 0,
              "nambpack: the pack region must be a whole number of erase sectors, "
              "or erasing it would reach into user data");

static_assert(FLASH_BASE % SECTOR_SIZE == 0, "nambpack: the pack must start on an erase-sector boundary");

static_assert(TOC_SLOT_SIZE % SECTOR_SIZE == 0, "nambpack: a TOC slot must be a whole number of erase sectors");

static_assert(sizeof(Header) + MAX_ENTRIES * sizeof(Entry) <= TOC_SLOT_SIZE,
              "nambpack: MAX_ENTRIES no longer fits in a TOC slot -- grow TOC_SLOT_SIZE");

static_assert(FIRST_BLOB_OFFSET < REGION_SIZE, "nambpack: the TOC area fills the whole pack region");

static_assert(HEADER_CRC_OFFSET + HEADER_CRC_SIZE <= sizeof(Header),
              "nambpack: the header checksum field must lie inside the header");

// =============================================================================
// Checksums
// =============================================================================

/// \brief CRC32 over a header with its own checksum field excluded.
///
/// The header carries count, total_size and above all sequence, and sequence is
/// what decides which slot the firmware believes. A half-programmed header with
/// a plausible sequence would otherwise win the arbitration and hand out entry
/// offsets that were never written, so the header is checksummed independently
/// of the TOC it introduces.
inline uint32_t HeaderCrc32(const Header& h)
{
  const uint8_t* p = reinterpret_cast<const uint8_t*>(&h);
  uint32_t crc = 0xFFFFFFFFu;
  for (size_t i = 0; i < sizeof(Header); i++)
  {
    if (i >= HEADER_CRC_OFFSET && i < HEADER_CRC_OFFSET + HEADER_CRC_SIZE)
      continue; // Skip the checksum field itself
    crc = nam::namb::crc32_table(static_cast<uint8_t>(crc ^ p[i])) ^ (crc >> 8);
  }
  return crc ^ 0xFFFFFFFFu;
}

/// \brief CRC32 over the live part of a TOC.
///
/// \param slot  Start of the TOC slot -- its Header, not its first Entry.
/// \param count Entries actually in use. Records past \p count are erased flash
///              and are never read, so hashing them would only make the
///              checksum depend on bytes nothing else cares about.
inline uint32_t TocCrc32(const uint8_t* slot, uint16_t count)
{
  return nam::namb::crc32(slot + sizeof(Header), static_cast<size_t>(count) * sizeof(Entry));
}

/// \brief Sequence comparison that survives wraparound.
///
/// Sequences advance by one per commit, so a signed difference is the right
/// test even though reaching 2^32 commits would outlast the flash by a wide
/// margin. Equal sequences are not "newer": the reader breaks that tie by slot
/// order so two identical slots resolve the same way every boot.
inline bool SequenceNewer(uint32_t a, uint32_t b)
{
  return static_cast<int32_t>(a - b) > 0;
}

} // namespace nambpack
} // namespace nam
