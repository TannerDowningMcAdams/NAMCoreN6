#pragma once
// Model pack (.nambpack): a table of contents plus N .namb blobs, programmed to
// external flash as one image and read in place.
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
//               |  user data   (24 MiB)    |  <- logs, recordings, measurements
//   0x92000000  +--------------------------+
//
// Everything is little-endian, matching the .namb container and the target.

#include <cstddef>
#include <cstdint>

namespace nam
{
namespace nambpack
{

/// \brief Magic number: "NMPK" as a little-endian uint32.
static constexpr uint32_t MAGIC = 0x4B504D4Eu;

/// \brief Layout version. Bump on any incompatible change to the structures below.
static constexpr uint16_t FORMAT_VERSION = 1;

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

/// \brief Upper bound on entries. Keeps the TOC a fixed, bounded read for the
///        firmware and keeps the whole thing well inside one sector.
static constexpr uint16_t MAX_ENTRIES = 64;

/// \brief Characters reserved for an entry name, including the NUL.
static constexpr size_t NAME_SIZE = 32;

#pragma pack(push, 1)

/// \brief Fixed 32-byte header at the start of the pack.
struct Header
{
  uint32_t magic; ///< MAGIC
  uint16_t version; ///< FORMAT_VERSION
  uint16_t count; ///< Number of valid Entry records that follow
  uint32_t total_size; ///< Bytes of the whole pack image, header included
  uint32_t toc_crc32; ///< CRC32 over count * sizeof(Entry) bytes of TOC
  uint32_t reserved[4]; ///< Zero
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

static_assert(REGION_SIZE % SECTOR_SIZE == 0,
              "nambpack: the pack region must be a whole number of erase sectors, "
              "or erasing it would reach into user data");

static_assert(FLASH_BASE % SECTOR_SIZE == 0, "nambpack: the pack must start on an erase-sector boundary");

/// \brief Byte offset at which the first blob may begin, for \p count entries.
///        The TOC is padded out to a sector so blobs stay sector-aligned.
inline constexpr uint32_t FirstBlobOffset(uint16_t count)
{
  return ((static_cast<uint32_t>(sizeof(Header)) + count * static_cast<uint32_t>(sizeof(Entry)) + SECTOR_SIZE - 1u)
          / SECTOR_SIZE)
         * SECTOR_SIZE;
}

} // namespace nambpack
} // namespace nam
