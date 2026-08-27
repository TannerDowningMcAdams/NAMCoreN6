// nambpack: combine .namb models into one flash image for the model-pack region.
//
// Usage:
//   nambpack -o modelpack.bin model1.namb [model2.namb ...]
//   nambpack -l modelpack.bin                        (list an existing pack)
//
// Options:
//   -o <file>     Output image. Required when packing.
//   -l <file>     List the contents of a pack and verify it, then exit.
//   -n <name>     Name for the next input file (default: its stem, truncated).
//   --align <n>   Blob alignment in bytes (default: the 4096-byte erase sector).
//   --no-verify   Skip per-model .namb header/CRC validation.
//
// Program the result with the external loader:
//   STM32_Programmer_CLI -c port=SWD -el <ExtMemLoader>.stldr \
//                        -d modelpack.bin 0x90200000 -v
//
// The tool refuses to emit an image larger than the pack region, because the
// bytes past it are user data -- logs, recordings, measurements -- and a
// too-large image would be programmed straight over them.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <namb/namb_format.h>
#include <namb/nambpack_format.h>

namespace
{

using nam::nambpack::Entry;
using nam::nambpack::Header;

struct Input
{
  std::filesystem::path path;
  std::string name;
  std::vector<uint8_t> data;
};

// ----------------------------------------------------------------------------

void usage()
{
  std::fprintf(stderr,
               "Usage: nambpack -o <out.bin> [-n <name>] <model.namb> [...]\n"
               "       nambpack -l <pack.bin>\n"
               "\n"
               "  -o <file>     output image\n"
               "  -l <file>     list and verify an existing pack, then exit\n"
               "  -n <name>     name for the next input (default: file stem)\n"
               "  --align <n>   blob alignment, bytes (default %u)\n"
               "  --no-verify   skip per-model .namb validation\n",
               (unsigned)nam::nambpack::SECTOR_SIZE);
}

bool read_file(const std::filesystem::path& p, std::vector<uint8_t>& out)
{
  std::ifstream f(p, std::ios::binary | std::ios::ate);
  if (!f.is_open())
    return false;
  const std::streamoff n = f.tellg();
  if (n < 0)
    return false;
  f.seekg(0, std::ios::beg);
  out.resize(static_cast<size_t>(n));
  if (!out.empty())
    f.read(reinterpret_cast<char*>(out.data()), static_cast<std::streamsize>(out.size()));
  return f.good() || f.eof();
}

// Validate a .namb the same way the firmware loader will: magic, format version,
// declared size, CRC32. Catching a bad model here is the difference between a
// tool error and a pedal that refuses to load a preset in the field.
bool verify_namb(const Input& in, std::string& why)
{
  using namespace nam::namb;

  if (in.data.size() < FILE_HEADER_SIZE + METADATA_BLOCK_SIZE)
  {
    why = "shorter than the .namb header";
    return false;
  }

  uint32_t magic = 0;
  std::memcpy(&magic, in.data.data(), 4);
  if (magic != MAGIC)
  {
    why = "bad magic (not a .namb file)";
    return false;
  }

  uint16_t version = 0;
  std::memcpy(&version, in.data.data() + 4, 2);
  if (version < MIN_FORMAT_VERSION || version > FORMAT_VERSION)
  {
    why = "unsupported .namb format version " + std::to_string(version);
    return false;
  }

  uint32_t total = 0;
  std::memcpy(&total, in.data.data() + 8, 4);
  if (total > in.data.size())
  {
    why = "truncated: header declares " + std::to_string(total) + " bytes, file has "
          + std::to_string(in.data.size());
    return false;
  }

  uint32_t stored = 0;
  std::memcpy(&stored, in.data.data() + 24, 4);
  if (compute_file_crc32(in.data.data(), total) != stored)
  {
    why = "checksum mismatch";
    return false;
  }

  return true;
}

uint32_t align_up(uint32_t v, uint32_t a)
{
  return ((v + a - 1u) / a) * a;
}

// ----------------------------------------------------------------------------

int list_pack(const std::filesystem::path& path)
{
  using namespace nam::nambpack;

  std::vector<uint8_t> img;
  if (!read_file(path, img))
  {
    std::fprintf(stderr, "nambpack: cannot read %s\n", path.string().c_str());
    return 1;
  }
  if (img.size() < sizeof(Header))
  {
    std::fprintf(stderr, "nambpack: %s is too small to be a pack\n", path.string().c_str());
    return 1;
  }

  Header h{};
  std::memcpy(&h, img.data(), sizeof(h));

  if (h.magic != MAGIC)
  {
    std::fprintf(stderr, "nambpack: bad magic (not a pack image)\n");
    return 1;
  }
  if (h.version != FORMAT_VERSION)
  {
    std::fprintf(stderr, "nambpack: unsupported pack version %u\n", (unsigned)h.version);
    return 1;
  }
  if (h.count > MAX_ENTRIES)
  {
    std::fprintf(stderr, "nambpack: entry count %u exceeds MAX_ENTRIES\n", (unsigned)h.count);
    return 1;
  }
  if (h.total_size > img.size())
  {
    std::fprintf(stderr, "nambpack: truncated (header says %u bytes, file has %zu)\n", h.total_size, img.size());
    return 1;
  }

  const size_t toc_bytes = static_cast<size_t>(h.count) * sizeof(Entry);
  if (sizeof(Header) + toc_bytes > img.size())
  {
    std::fprintf(stderr, "nambpack: TOC extends past end of file\n");
    return 1;
  }
  const uint32_t toc_crc = nam::namb::crc32(img.data() + sizeof(Header), toc_bytes);
  if (toc_crc != h.toc_crc32)
  {
    std::fprintf(stderr, "nambpack: TOC checksum mismatch\n");
    return 1;
  }

  std::printf("pack %s\n", path.string().c_str());
  std::printf("  version %u, %u model(s), %u bytes (%.1f%% of the %u KiB region)\n\n", (unsigned)h.version,
              (unsigned)h.count, h.total_size, 100.0 * h.total_size / REGION_SIZE, REGION_SIZE / 1024u);
  std::printf("  %-3s %-32s %10s %10s  %s\n", "idx", "name", "offset", "size", "flash address");

  int bad = 0;
  for (uint16_t i = 0; i < h.count; i++)
  {
    Entry e{};
    std::memcpy(&e, img.data() + sizeof(Header) + i * sizeof(Entry), sizeof(e));
    char name[NAME_SIZE + 1];
    std::memcpy(name, e.name, NAME_SIZE);
    name[NAME_SIZE] = '\0';

    const bool in_range = (static_cast<uint64_t>(e.offset) + e.size) <= h.total_size;
    std::printf("  %-3u %-32s %10u %10u  0x%08X%s\n", (unsigned)i, name, e.offset, e.size, FLASH_BASE + e.offset,
                in_range ? "" : "   <-- OUT OF RANGE");
    if (!in_range)
      bad++;
  }

  if (bad)
  {
    std::fprintf(stderr, "\nnambpack: %d entry/entries out of range\n", bad);
    return 1;
  }
  std::printf("\n  ok\n");
  return 0;
}

} // namespace

// ----------------------------------------------------------------------------

int main(int argc, char** argv)
{
  using namespace nam::nambpack;

  std::filesystem::path out_path;
  std::vector<Input> inputs;
  std::string pending_name;
  uint32_t alignment = SECTOR_SIZE;
  bool verify = true;

  for (int i = 1; i < argc; i++)
  {
    const std::string a = argv[i];

    if (a == "-h" || a == "--help")
    {
      usage();
      return 0;
    }
    if (a == "-l")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      return list_pack(argv[i]);
    }
    if (a == "-o")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      out_path = argv[i];
      continue;
    }
    if (a == "-n")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      pending_name = argv[i];
      continue;
    }
    if (a == "--align")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      alignment = static_cast<uint32_t>(std::stoul(argv[i]));
      if (alignment == 0 || (alignment & (alignment - 1)) != 0)
      {
        std::fprintf(stderr, "nambpack: --align must be a power of two\n");
        return 1;
      }
      continue;
    }
    if (a == "--no-verify")
    {
      verify = false;
      continue;
    }
    if (!a.empty() && a[0] == '-')
    {
      std::fprintf(stderr, "nambpack: unknown option '%s'\n", a.c_str());
      usage();
      return 1;
    }

    Input in;
    in.path = a;
    in.name = pending_name.empty() ? in.path.stem().string() : pending_name;
    pending_name.clear();
    inputs.push_back(std::move(in));
  }

  if (out_path.empty() || inputs.empty())
  {
    usage();
    return 1;
  }
  if (inputs.size() > MAX_ENTRIES)
  {
    std::fprintf(stderr, "nambpack: %zu models exceeds MAX_ENTRIES (%u)\n", inputs.size(), (unsigned)MAX_ENTRIES);
    return 1;
  }

  // --- Read and validate every input before laying anything out --------------
  for (auto& in : inputs)
  {
    if (!read_file(in.path, in.data))
    {
      std::fprintf(stderr, "nambpack: cannot read %s\n", in.path.string().c_str());
      return 1;
    }
    if (in.data.empty())
    {
      std::fprintf(stderr, "nambpack: %s is empty\n", in.path.string().c_str());
      return 1;
    }
    if (in.name.size() >= NAME_SIZE)
    {
      std::fprintf(stderr, "nambpack: name '%s' is longer than %zu characters\n", in.name.c_str(), NAME_SIZE - 1);
      return 1;
    }
    if (verify)
    {
      std::string why;
      if (!verify_namb(in, why))
      {
        std::fprintf(stderr, "nambpack: %s is not a valid .namb: %s\n", in.path.string().c_str(), why.c_str());
        return 1;
      }
    }
  }

  // Duplicate names would make the firmware's lookup ambiguous.
  for (size_t i = 0; i < inputs.size(); i++)
  {
    for (size_t j = i + 1; j < inputs.size(); j++)
    {
      if (inputs[i].name == inputs[j].name)
      {
        std::fprintf(stderr, "nambpack: duplicate model name '%s'\n", inputs[i].name.c_str());
        return 1;
      }
    }
  }

  // --- Lay out -------------------------------------------------------------
  const uint16_t count = static_cast<uint16_t>(inputs.size());
  std::vector<Entry> toc(count);

  uint32_t cursor = align_up(static_cast<uint32_t>(sizeof(Header)) + count * static_cast<uint32_t>(sizeof(Entry)),
                             alignment);

  for (uint16_t i = 0; i < count; i++)
  {
    Entry& e = toc[i];
    std::memset(&e, 0, sizeof(e));
    e.offset = cursor;
    e.size = static_cast<uint32_t>(inputs[i].data.size());
    std::memcpy(e.name, inputs[i].name.c_str(), inputs[i].name.size());

    // Overflow-safe: cursor and size are both u32, so widen before adding.
    const uint64_t end = static_cast<uint64_t>(cursor) + e.size;
    if (end > REGION_SIZE)
    {
      std::fprintf(stderr,
                   "nambpack: '%s' would end at %llu bytes, past the %u KiB pack region.\n"
                   "          Programming this image would overwrite user data at 0x%08X.\n",
                   inputs[i].name.c_str(), (unsigned long long)end, REGION_SIZE / 1024u, FLASH_BASE + REGION_SIZE);
      return 1;
    }
    cursor = align_up(static_cast<uint32_t>(end), alignment);
  }

  // The final image is trimmed to the last blob rather than padded to the
  // alignment, so the programmer writes (and verifies) only real bytes.
  const uint32_t total_size = toc[count - 1].offset + toc[count - 1].size;

  if (total_size > REGION_SIZE)
  {
    std::fprintf(stderr, "nambpack: image is %u bytes, pack region is %u\n", total_size, REGION_SIZE);
    return 1;
  }

  // --- Emit ----------------------------------------------------------------
  std::vector<uint8_t> img(total_size, 0xFF); // 0xFF = erased flash, so padding
                                              // costs no extra program cycles

  Header h{};
  h.magic = MAGIC;
  h.version = FORMAT_VERSION;
  h.count = count;
  h.total_size = total_size;
  h.toc_crc32 = nam::namb::crc32(reinterpret_cast<const uint8_t*>(toc.data()), toc.size() * sizeof(Entry));

  std::memcpy(img.data(), &h, sizeof(h));
  std::memcpy(img.data() + sizeof(Header), toc.data(), toc.size() * sizeof(Entry));
  for (uint16_t i = 0; i < count; i++)
    std::memcpy(img.data() + toc[i].offset, inputs[i].data.data(), inputs[i].data.size());

  std::ofstream f(out_path, std::ios::binary | std::ios::trunc);
  if (!f.is_open())
  {
    std::fprintf(stderr, "nambpack: cannot write %s\n", out_path.string().c_str());
    return 1;
  }
  f.write(reinterpret_cast<const char*>(img.data()), static_cast<std::streamsize>(img.size()));
  if (!f.good())
  {
    std::fprintf(stderr, "nambpack: write failed\n");
    return 1;
  }
  f.close();

  // --- Report --------------------------------------------------------------
  std::printf("wrote %s\n", out_path.string().c_str());
  std::printf("  %-3s %-32s %10s %10s  %s\n", "idx", "name", "offset", "size", "flash address");
  for (uint16_t i = 0; i < count; i++)
  {
    std::printf("  %-3u %-32s %10u %10u  0x%08X\n", (unsigned)i, inputs[i].name.c_str(), toc[i].offset, toc[i].size,
                FLASH_BASE + toc[i].offset);
  }

  const uint32_t payload = [&] {
    uint32_t s = 0;
    for (const auto& in : inputs)
      s += static_cast<uint32_t>(in.data.size());
    return s;
  }();

  std::printf("\n  %u model(s), %u bytes total (%u payload, %u padding/TOC)\n", (unsigned)count, total_size, payload,
              total_size - payload);
  std::printf("  region 0x%08X..0x%08X (%u KiB), %.1f%% used, %u KiB free\n", FLASH_BASE, FLASH_BASE + REGION_SIZE,
              REGION_SIZE / 1024u, 100.0 * total_size / REGION_SIZE, (REGION_SIZE - total_size) / 1024u);
  std::printf("  user data begins at 0x%08X and is untouched\n", FLASH_BASE + REGION_SIZE);

  return 0;
}
