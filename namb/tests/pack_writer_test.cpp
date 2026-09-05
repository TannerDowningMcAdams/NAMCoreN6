// PackWriter against a simulated NOR.
//
// This is code that erases flash, so the simulator is deliberately hostile: it
// asserts on anything the real part would do something undefined about, and it
// makes the memory-mapped view unreadable while the map is suspended, so a read
// of flash inside the write window shows up as corrupt data rather than working
// by accident on the host.
//
// What it checks that inspection cannot: that a commit never touches a sector
// outside the pack region, that first-fit reuses freed space, and that a power
// loss at any step leaves the previous pack intact -- the property the whole
// two-slot design exists for.

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <namb/pack_writer.h>

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

// ---------------------------------------------------------------------------
// Simulated W25Q
// ---------------------------------------------------------------------------

struct Nor
{
  static constexpr uint32_t kChipBase = FLASH_BASE - 0x90000000u;

  std::vector<uint8_t> store; ///< True contents
  std::vector<uint8_t> view; ///< What `base` points at; poisoned while suspended
  bool suspended = false;

  int erases = 0;
  int programs = 0;
  int ops_until_failure = -1; ///< -1 = never fail
  bool violated = false;
  const char* violation = nullptr;

  Nor()
  : store(REGION_SIZE, 0xFF)
  , view(REGION_SIZE, 0xFF)
  {
  }

  void fail(const char* why)
  {
    if (!violated)
    {
      violated = true;
      violation = why;
    }
  }

  bool budget()
  {
    if (ops_until_failure < 0)
      return true;
    if (ops_until_failure == 0)
      return false;
    ops_until_failure--;
    return true;
  }

  bool in_region(uint32_t chip_addr, uint32_t len)
  {
    return chip_addr >= kChipBase && (static_cast<uint64_t>(chip_addr) + len) <= (kChipBase + REGION_SIZE);
  }

  // --- ops -----------------------------------------------------------------

  static bool Suspend(void* c)
  {
    Nor& n = *static_cast<Nor*>(c);
    if (n.suspended)
      n.fail("suspend while already suspended");
    n.suspended = true;
    std::memset(n.view.data(), 0xA5, n.view.size()); // reads here are a bug
    return true;
  }

  static bool Resume(void* c)
  {
    Nor& n = *static_cast<Nor*>(c);
    if (!n.suspended)
      n.fail("resume without suspend");
    n.suspended = false;
    n.view = n.store;
    return true;
  }

  static bool EraseSector(void* c, uint32_t sector)
  {
    Nor& n = *static_cast<Nor*>(c);
    const uint32_t addr = sector * SECTOR_SIZE;

    if (!n.suspended)
      n.fail("erase with the memory map live");
    if (!n.in_region(addr, SECTOR_SIZE))
    {
      // Would eat user data at 0x90800000 on the real part. Record and stop --
      // going on would run off the simulated array and crash instead of report.
      n.fail("erase outside the pack region");
      return false;
    }
    if (!n.budget())
    {
      // Model a power cut part-way: the sector is gone, the commit is not.
      if (n.in_region(addr, SECTOR_SIZE))
        std::memset(n.store.data() + (addr - kChipBase), 0xFF, SECTOR_SIZE);
      return false;
    }

    n.erases++;
    std::memset(n.store.data() + (addr - kChipBase), 0xFF, SECTOR_SIZE);
    return true;
  }

  static bool Program(void* c, uint32_t addr, const uint8_t* src, uint16_t len)
  {
    Nor& n = *static_cast<Nor*>(c);

    if (!n.suspended)
      n.fail("program with the memory map live");
    if (len == 0 || len > 256)
      n.fail("program length outside 1..256");
    if ((addr % 256u) + len > 256u)
      n.fail("page program crossing a page boundary"); // the part would wrap, not carry
    if (!n.in_region(addr, len))
    {
      n.fail("program outside the pack region");
      return false;
    }
    if (!n.budget())
      return false;

    n.programs++;
    uint8_t* dst = n.store.data() + (addr - kChipBase);
    for (uint16_t i = 0; i < len; i++)
      dst[i] &= src[i]; // NOR clears bits; it cannot set them
    return true;
  }

  static void Invalidate(void*, const uint8_t*, uint32_t) {}

  FlashOps ops()
  {
    FlashOps o;
    o.base = view.data();
    o.chip_base = kChipBase;
    o.ctx = this;
    o.suspend = &Suspend;
    o.resume = &Resume;
    o.erase_sector = &EraseSector;
    o.program = &Program;
    o.invalidate = &Invalidate;
    return o;
  }
};

/// A blob that looks enough like a .namb to be stored, with a per-model filler
/// so a mixed-up entry shows as a content mismatch rather than a size one.
std::vector<uint8_t> make_blob(uint32_t size, uint8_t tag)
{
  std::vector<uint8_t> b(size, tag);
  const uint32_t magic = nam::namb::MAGIC;
  std::memcpy(b.data(), &magic, 4);
  b[4] = tag;
  return b;
}

bool stored_matches(Nor& nor, const char* name, const std::vector<uint8_t>& blob)
{
  PackView p;
  if (!IsOk(p.Open(nor.view.data(), REGION_SIZE)))
    return false;
  const Entry* e = p.Find(name);
  if (e == nullptr || e->size != blob.size())
    return false;
  return std::memcmp(p.Blob(*e), blob.data(), blob.size()) == 0;
}

uint16_t count_of(Nor& nor)
{
  PackView p;
  return IsOk(p.Open(nor.view.data(), REGION_SIZE)) ? p.Count() : 0;
}

} // namespace

int main()
{
  const std::vector<uint8_t> a = make_blob(8028, 0xA1);
  const std::vector<uint8_t> b = make_blob(8028, 0xB2);
  const std::vector<uint8_t> c = make_blob(12000, 0xC3);

  std::printf("format and add\n");
  Nor nor;
  {
    PackWriter w(nor.ops());
    check(IsOk(w.Format()), "format an erased chip");
    check(count_of(nor) == 0, "empty pack");

    check(IsOk(w.Add("alpha", a.data(), static_cast<uint32_t>(a.size()))), "add alpha");
    check(IsOk(w.Add("beta", b.data(), static_cast<uint32_t>(b.size()))), "add beta");
    check(IsOk(w.Add("gamma", c.data(), static_cast<uint32_t>(c.size()))), "add gamma");
    check(count_of(nor) == 3, "three models");
    check(stored_matches(nor, "alpha", a), "alpha reads back");
    check(stored_matches(nor, "beta", b), "beta reads back");
    check(stored_matches(nor, "gamma", c), "gamma reads back");

    PackView p;
    p.Open(nor.view.data(), REGION_SIZE);
    check(p.Sequence() == 4, "one sequence step per commit");
    check(p.At(0)->offset == FIRST_BLOB_OFFSET, "first blob at FIRST_BLOB_OFFSET");
    check(p.At(1)->offset >= p.At(0)->offset + p.At(0)->size, "blobs do not overlap");
    check(p.At(1)->offset % SECTOR_SIZE == 0, "blobs are sector aligned");
  }

  std::printf("\nslot alternation\n");
  {
    PackView p;
    p.Open(nor.view.data(), REGION_SIZE);
    const uint8_t before = p.ActiveSlot();

    PackWriter w(nor.ops());
    check(IsOk(w.Add("delta", a.data(), static_cast<uint32_t>(a.size()))), "add delta");

    PackView q;
    q.Open(nor.view.data(), REGION_SIZE);
    check(q.ActiveSlot() != before, "the commit landed in the other slot");
    check(q.Sequence() == p.Sequence() + 1, "sequence advanced by one");
  }

  std::printf("\nreplace and remove\n");
  {
    const std::vector<uint8_t> bigger = make_blob(20000, 0xD4);
    PackWriter w(nor.ops());

    check(IsOk(w.Add("beta", bigger.data(), static_cast<uint32_t>(bigger.size()))), "replace beta, larger");
    check(count_of(nor) == 4, "count unchanged by a replace");
    check(stored_matches(nor, "beta", bigger), "beta is the new content");

    check(IsOk(w.Remove("gamma")), "remove gamma");
    check(count_of(nor) == 3, "count dropped");
    {
      PackView p;
      p.Open(nor.view.data(), REGION_SIZE);
      check(p.Find("gamma") == nullptr, "gamma is gone");
      check(p.Find("alpha") != nullptr, "alpha survived");
    }
    check(w.Remove("gamma") == Status::ErrorInvalidConfig, "removing it twice is refused");
  }

  std::printf("\nfree space reuse\n");
  {
    PackWriter w(nor.ops());
    PackInfo before{};
    check(IsOk(w.Query(before)), "query");

    // gamma's 12000 bytes (3 sectors) and beta's old 8028 (2) are both free
    // now, and both sit below the last blob -- so first-fit must go backwards
    // rather than appending.
    PackView p;
    p.Open(nor.view.data(), REGION_SIZE);
    uint32_t highest = 0;
    for (uint16_t i = 0; i < p.Count(); i++)
      if (p.At(i)->offset > highest)
        highest = p.At(i)->offset;

    const std::vector<uint8_t> small = make_blob(4000, 0xE5);
    check(IsOk(w.Add("epsilon", small.data(), static_cast<uint32_t>(small.size()))), "add epsilon");

    PackView q;
    q.Open(nor.view.data(), REGION_SIZE);
    check(q.Find("epsilon")->offset < highest, "epsilon reused a hole rather than appending");
    check(stored_matches(nor, "epsilon", small), "epsilon reads back");
    check(stored_matches(nor, "alpha", a), "alpha still intact after the reuse");

    PackInfo after{};
    w.Query(after);
    check(after.count == before.count + 1, "query tracks the count");
    check(after.largest_free > 0 && after.largest_free < REGION_SIZE, "largest_free is plausible");
  }

  std::printf("\nlimits\n");
  {
    PackWriter w(nor.ops());
    const std::vector<uint8_t> huge = make_blob(REGION_SIZE - FIRST_BLOB_OFFSET, 0xF6);
    check(w.Add("huge", huge.data(), static_cast<uint32_t>(huge.size())) == Status::ErrorTooSmall,
          "a blob with nowhere to fit is refused");
    check(stored_matches(nor, "alpha", a), "and the pack is untouched");

    check(w.Add("", a.data(), 4) == Status::ErrorInvalidConfig, "an empty name is refused");
    check(w.Add("x", a.data(), 0) == Status::ErrorInvalidConfig, "a zero-length blob is refused");
    check(w.Add("x", nullptr, 4) == Status::ErrorInvalidConfig, "a null blob is refused");
  }

  std::printf("\nMAX_ENTRIES\n");
  {
    Nor full;
    PackWriter w(full.ops());
    w.Format();

    const std::vector<uint8_t> tiny = make_blob(64, 0x11);
    bool all_ok = true;
    for (int i = 0; i < MAX_ENTRIES; i++)
    {
      char name[NAME_SIZE];
      std::snprintf(name, sizeof(name), "m%03d", i);
      if (!IsOk(w.Add(name, tiny.data(), static_cast<uint32_t>(tiny.size()))))
        all_ok = false;
    }
    check(all_ok, "MAX_ENTRIES models fit");
    check(count_of(full) == MAX_ENTRIES, "and all are present");
    check(w.Add("one_too_many", tiny.data(), 64) == Status::ErrorInvalidConfig, "the next one is refused");
    check(!full.violated, full.violated ? full.violation : "no simulator violations");
  }

  std::printf("\npower loss\n");
  {
    // Snapshot a good pack, then cut power at each step of the next commit and
    // demand the pack still reads as it did before.
    Nor base;
    {
      PackWriter w(base.ops());
      w.Format();
      w.Add("alpha", a.data(), static_cast<uint32_t>(a.size()));
      w.Add("beta", b.data(), static_cast<uint32_t>(b.size()));
    }

    PackView before;
    before.Open(base.view.data(), REGION_SIZE);
    const uint32_t good_sequence = before.Sequence();
    const uint16_t good_count = before.Count();

    int survived = 0;
    int attempted = 0;
    int rolled_back = 0;
    int completed = 0;
    // Wide enough to run past the end of a commit, so the sweep covers both
    // sides of the point where the new TOC becomes live.
    for (int cut = 0; cut < 80; cut++)
    {
      Nor n;
      n.store = base.store;
      n.view = base.view;
      n.ops_until_failure = cut;

      PackWriter w(n.ops());
      const Status st = w.Add("gamma", c.data(), static_cast<uint32_t>(c.size()));
      attempted++;

      PackView p;
      const Status opened = p.Open(n.view.data(), REGION_SIZE);
      if (!IsOk(opened))
        continue; // counted as a survivor only if it opens

      // Either the commit landed whole, or the pack is exactly as it was.
      const bool committed = IsOk(st) && p.Find("gamma") != nullptr;
      const bool unchanged = p.Count() == good_count && p.Sequence() == good_sequence
                             && p.Find("alpha") != nullptr && p.Find("beta") != nullptr
                             && std::memcmp(p.Blob(*p.Find("alpha")), a.data(), a.size()) == 0;

      if (committed || unchanged)
        survived++;
      else
        std::printf("        cut at op %d left the pack in neither state\n", cut);

      completed += committed ? 1 : 0;
      rolled_back += (!committed && unchanged) ? 1 : 0;

      if (n.violated)
        std::printf("        cut at op %d: %s\n", cut, n.violation);
    }
    check(survived == attempted, "every power cut leaves the old pack or the new one, never neither");
    // Without these two the loop could pass by never cutting anything, or by
    // never reaching the far side of the commit.
    check(rolled_back > 0, "the sweep interrupted some commits");
    check(completed > 0, "and ran past the end of others");
    std::printf("        %d of %d cuts rolled back, %d completed\n", rolled_back, attempted, completed);
  }

  std::printf("\nsimulator self-check\n");
  {
    // The assertions above are only worth anything if they can fire.
    Nor probe;
    Nor::Suspend(&probe);
    Nor::Program(&probe, Nor::kChipBase + 200, a.data(), 100); // 200+100 crosses 256
    check(probe.violated && std::strstr(probe.violation, "page") != nullptr, "a page-crossing program is caught");

    Nor probe2;
    Nor::Suspend(&probe2);
    Nor::EraseSector(&probe2, (Nor::kChipBase + REGION_SIZE) / SECTOR_SIZE); // first user-data sector
    check(probe2.violated && std::strstr(probe2.violation, "region") != nullptr,
          "an erase past the pack region is caught");

    Nor probe3;
    Nor::Program(&probe3, Nor::kChipBase, a.data(), 16); // no suspend
    check(probe3.violated && std::strstr(probe3.violation, "live") != nullptr,
          "a write with the memory map live is caught");
  }

  std::printf("\nsimulator\n");
  check(!nor.violated, nor.violated ? nor.violation : "no page-crossing, out-of-region or live-map access");
  check(nor.erases > 0 && nor.programs > 0, "the flash was actually written");
  std::printf("        %d erases, %d page programs\n", nor.erases, nor.programs);

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
