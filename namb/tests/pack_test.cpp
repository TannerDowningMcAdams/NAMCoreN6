// PackView / nambpack format v2 conformance.
//
// Builds a pack image the way nambpack lays one out, then checks the reader
// agrees about it -- and, more importantly, that the two-slot arbitration
// behaves under the failures it exists for: a torn header, a torn TOC, a
// sequence that has wrapped, and two slots that somehow claim the same one.
//
// Self-contained by default. Pass a real pack image to validate that instead:
//
//   pack_test [modelpack.bin]

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <namb/nambpack_reader.h>

using namespace nam;
using namespace nam::nambpack;

namespace
{

int failures = 0;

void check(bool cond, const char* what)
{
  std::printf("  %-58s %s\n", what, cond ? "ok" : "FAIL");
  if (!cond)
    failures++;
}

uint32_t align_up(uint32_t v, uint32_t a)
{
  return ((v + a - 1u) / a) * a;
}

/// A pack laid out exactly as nambpack emits one: slot A at sequence 1, slot B
/// erased, blobs sector-aligned from FIRST_BLOB_OFFSET. The blobs carry a .namb
/// magic so Blob() can be checked, but are not otherwise real models -- PackView
/// deliberately does not look inside them.
std::vector<uint8_t> synth_pack(uint16_t count, uint32_t blob_size)
{
  std::vector<Entry> toc(count);
  uint32_t cursor = FIRST_BLOB_OFFSET;
  for (uint16_t i = 0; i < count; i++)
  {
    std::memset(&toc[i], 0, sizeof(Entry));
    toc[i].offset = cursor;
    toc[i].size = blob_size;
    std::snprintf(toc[i].name, NAME_SIZE, "synthetic_model_%u", static_cast<unsigned>(i));
    cursor = align_up(cursor + blob_size, SECTOR_SIZE);
  }

  const uint32_t total = toc[count - 1].offset + blob_size;
  std::vector<uint8_t> img(total, 0xFF);

  for (uint16_t i = 0; i < count; i++)
  {
    const uint32_t magic = nam::namb::MAGIC;
    std::memcpy(img.data() + toc[i].offset, &magic, 4);
  }

  const uint32_t slot_a = TocSlotOffset(0);
  std::memcpy(img.data() + slot_a + sizeof(Header), toc.data(), toc.size() * sizeof(Entry));

  Header h{};
  h.magic = MAGIC;
  h.version = FORMAT_VERSION;
  h.count = count;
  h.total_size = total;
  h.toc_crc32 = TocCrc32(img.data() + slot_a, count);
  h.sequence = 1;
  h.header_crc32 = HeaderCrc32(h);
  std::memcpy(img.data() + slot_a, &h, sizeof(h));

  return img;
}

std::vector<uint8_t> load(const char* path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
  {
    std::fprintf(stderr, "pack_test: cannot open %s\n", path);
    std::exit(2);
  }
  const std::streamoff n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> v(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

Header* hdr(std::vector<uint8_t>& img, uint8_t slot)
{
  return reinterpret_cast<Header*>(img.data() + TocSlotOffset(slot));
}

/// Copy one slot's header and TOC onto another, restamped with \p seq. This is
/// what a commit does, minus the flash.
void clone_slot(std::vector<uint8_t>& img, uint8_t from, uint8_t to, uint32_t seq)
{
  std::memcpy(img.data() + TocSlotOffset(to), img.data() + TocSlotOffset(from), TOC_SLOT_SIZE);
  Header* h = hdr(img, to);
  h->sequence = seq;
  h->header_crc32 = HeaderCrc32(*h);
}

} // namespace

int main(int argc, char** argv)
{
  const std::vector<uint8_t> golden = (argc > 1) ? load(argv[1]) : synth_pack(4, 8028);
  const uint32_t size = static_cast<uint32_t>(golden.size());
  const uint16_t expect_count = (argc > 1) ? 0 : 4; // 0 = "whatever the file holds"

  std::printf("%s\n", (argc > 1) ? argv[1] : "synthetic pack (4 models)");

  std::printf("\nlayout\n");
  check(TOC_SLOT_SIZE % SECTOR_SIZE == 0, "a TOC slot is a whole number of erase sectors");
  check(FIRST_BLOB_OFFSET == TOC_SLOT_COUNT * TOC_SLOT_SIZE, "blobs start past every TOC slot");
  check(sizeof(Header) + MAX_ENTRIES * sizeof(Entry) <= TOC_SLOT_SIZE, "MAX_ENTRIES fits a slot");
  check(sizeof(Header) == 32 && sizeof(Entry) == 40, "struct sizes unchanged from v1");

  std::printf("\nas emitted by nambpack\n");
  {
    PackView p;
    const Status st = p.Open(golden.data(), size);
    check(IsOk(st), "opens");
    if (expect_count)
      check(p.Count() == expect_count, "entry count is what was packed");
    check(p.ActiveSlot() == 0, "slot A active");
    check(p.NextSlot() == 1, "next commit targets slot B");

    const Entry* e = p.At(0);
    check(e != nullptr, "At(0)");
    if (e != nullptr)
    {
      check(e->offset >= FIRST_BLOB_OFFSET, "blob clears the TOC area");
      uint32_t magic = 0;
      std::memcpy(&magic, p.Blob(*e), 4);
      check(magic == nam::namb::MAGIC, "Blob() lands on the .namb magic");
      check(p.Find(e->name) == e, "Find() by name returns the same entry");
    }
    check(p.At(p.Count()) == nullptr, "At() past the count returns null");
    check(p.Find("nope") == nullptr, "Find() of an absent name returns null");
  }

  std::printf("\nA/B arbitration\n");
  {
    std::vector<uint8_t> img = golden;
    clone_slot(img, 0, 1, 2);
    PackView p;
    check(IsOk(p.Open(img.data(), size)), "opens with both slots valid");
    check(p.ActiveSlot() == 1, "higher sequence wins (B)");
    check(p.Sequence() == 2, "reports B's sequence");
    check(p.NextSlot() == 0, "next commit targets A");
  }
  {
    // A power loss part-way through a commit: slot B's header is inconsistent.
    std::vector<uint8_t> img = golden;
    clone_slot(img, 0, 1, 2);
    hdr(img, 1)->count = 99; // plausible-looking damage; the CRC no longer matches
    PackView p;
    check(IsOk(p.Open(img.data(), size)), "opens with B's header damaged");
    check(p.ActiveSlot() == 0, "falls back to A");
    check(p.Sequence() == 1, "and to A's sequence");
  }
  {
    // Damage confined to the TOC rather than the header.
    std::vector<uint8_t> img = golden;
    clone_slot(img, 0, 1, 2);
    img[TocSlotOffset(1) + sizeof(Header) + 4] ^= 0xFF;
    PackView p;
    check(IsOk(p.Open(img.data(), size)), "opens with B's TOC corrupt");
    check(p.ActiveSlot() == 0, "falls back to A");
  }
  {
    // Sequence wraparound: B at 0 is newer than A at 0xFFFFFFFF.
    std::vector<uint8_t> img = golden;
    hdr(img, 0)->sequence = 0xFFFFFFFFu;
    hdr(img, 0)->header_crc32 = HeaderCrc32(*hdr(img, 0));
    clone_slot(img, 0, 1, 0);
    PackView p;
    check(IsOk(p.Open(img.data(), size)), "opens across the wrap");
    check(p.ActiveSlot() == 1, "B (seq 0) beats A (seq 0xFFFFFFFF)");
  }
  {
    // Two slots claiming the same sequence must resolve the same way every boot.
    std::vector<uint8_t> img = golden;
    clone_slot(img, 0, 1, 1);
    PackView p;
    check(IsOk(p.Open(img.data(), size)), "opens with tied sequences");
    check(p.ActiveSlot() == 0, "tie goes to the lower slot index");
  }

  std::printf("\nrejection\n");
  {
    std::vector<uint8_t> img(size, 0xFF);
    PackView p;
    check(p.Open(img.data(), size) == Status::ErrorBadMagic, "erased flash -> ErrorBadMagic");
    check(!p.IsOpen(), "view stays closed");
    check(p.At(0) == nullptr, "At() on a closed view returns null");
    check(p.Find("x") == nullptr, "Find() on a closed view returns null");
  }
  {
    std::vector<uint8_t> img = golden;
    hdr(img, 0)->version = 1;
    hdr(img, 0)->header_crc32 = HeaderCrc32(*hdr(img, 0));
    PackView p;
    check(p.Open(img.data(), size) == Status::ErrorUnsupportedVersion, "a v1 pack -> ErrorUnsupportedVersion");
  }
  {
    std::vector<uint8_t> img = golden;
    img[TocSlotOffset(0) + sizeof(Header)] ^= 0xFF;
    PackView p;
    check(p.Open(img.data(), size) == Status::ErrorChecksum, "corrupt TOC -> ErrorChecksum");
  }
  {
    std::vector<uint8_t> img = golden;
    hdr(img, 0)->sequence ^= 0x5A5A5A5Au; // header changed, checksum not recomputed
    PackView p;
    check(p.Open(img.data(), size) == Status::ErrorChecksum, "corrupt header -> ErrorChecksum");
  }

  // Entry-level range checks. Each mutates one field and repairs both
  // checksums, so what is being tested is the range check and not the CRC.
  struct EntryCase
  {
    const char* what;
    Status expect;
    void (*mutate)(Entry&, uint32_t size);
  };
  const EntryCase entry_cases[] = {
    {"blob inside the TOC area -> ErrorInvalidConfig", Status::ErrorInvalidConfig,
     [](Entry& e, uint32_t) { e.offset = 2048; }},
    {"blob past total_size -> out of range", Status::ErrorWeightsOutOfRange,
     [](Entry& e, uint32_t s) { e.size = s; }},
    {"zero-length blob -> ErrorInvalidConfig", Status::ErrorInvalidConfig, [](Entry& e, uint32_t) { e.size = 0; }},
    {"unterminated name -> ErrorInvalidConfig", Status::ErrorInvalidConfig,
     [](Entry& e, uint32_t) { e.name[NAME_SIZE - 1] = 'x'; }},
  };

  for (const EntryCase& c : entry_cases)
  {
    std::vector<uint8_t> img = golden;
    Entry* e = reinterpret_cast<Entry*>(img.data() + TocSlotOffset(0) + sizeof(Header));
    c.mutate(*e, size);
    Header* h = hdr(img, 0);
    h->toc_crc32 = TocCrc32(img.data() + TocSlotOffset(0), h->count);
    h->header_crc32 = HeaderCrc32(*h);
    PackView p;
    check(p.Open(img.data(), size) == c.expect, c.what);
  }

  {
    PackView p;
    check(p.Open(golden.data(), FIRST_BLOB_OFFSET - 1) == Status::ErrorTooSmall,
          "a region below the TOC area -> ErrorTooSmall");
    check(p.Open(nullptr, size) == Status::ErrorTooSmall, "a null base -> ErrorTooSmall");
  }

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
