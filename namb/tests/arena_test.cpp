// Measures what an A2 .nam actually costs to parse through the arena, and
// checks the arena DOM agrees with a stock nlohmann one.
//
// The global operator new is counted so the split between "arena" and "still on
// the heap" (std::string keys and values) is visible rather than assumed.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <namb/json_arena.h>

using nam::json_arena::Arena;
using nam::json_arena::arena_json;
using nam::json_arena::ScopedBind;

// ---------------------------------------------------------------------------
// Global heap accounting
// ---------------------------------------------------------------------------

namespace
{
size_t g_heap_live = 0;
size_t g_heap_peak = 0;
size_t g_heap_total = 0;
size_t g_heap_count = 0;
bool g_counting = false;
int failures = 0;
} // namespace

void* operator new(size_t n)
{
  // Header carries the size so delete can subtract it. 16 bytes keeps the
  // returned pointer maximally aligned.
  void* raw = std::malloc(n + 16);
  if (raw == nullptr)
    std::abort();
  *static_cast<size_t*>(raw) = n;
  if (g_counting)
  {
    g_heap_live += n;
    g_heap_total += n;
    g_heap_count++;
    if (g_heap_live > g_heap_peak)
      g_heap_peak = g_heap_live;
  }
  return static_cast<char*>(raw) + 16;
}

void operator delete(void* p) noexcept
{
  if (p == nullptr)
    return;
  void* raw = static_cast<char*>(p) - 16;
  if (g_counting)
    g_heap_live -= *static_cast<size_t*>(raw);
  std::free(raw);
}

void operator delete(void* p, size_t) noexcept
{
  operator delete(p);
}

namespace
{

void check(bool cond, const char* what)
{
  std::printf("    %-52s %s\n", what, cond ? "ok" : "FAIL");
  if (!cond)
    failures++;
}

std::vector<char> load(const std::string& path)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
  {
    std::fprintf(stderr, "cannot open %s\n", path.c_str());
    std::exit(2);
  }
  const std::streamoff n = f.tellg();
  f.seekg(0);
  std::vector<char> v(static_cast<size_t>(n));
  f.read(v.data(), n);
  return v;
}

double kib(size_t b)
{
  return static_cast<double>(b) / 1024.0;
}

// Total weights in a document, counting a container's submodels.
size_t weight_total(const nlohmann::json& j)
{
  size_t n = 0;
  if (j.contains("weights") && j["weights"].is_array())
    n += j["weights"].size();
  if (j.contains("config") && j["config"].contains("submodels"))
  {
    for (const auto& sm : j["config"]["submodels"])
      if (sm.contains("model"))
        n += weight_total(sm["model"]);
  }
  return n;
}

size_t weight_total(const arena_json& j)
{
  size_t n = 0;
  if (j.contains("weights") && j["weights"].is_array())
    n += j["weights"].size();
  if (j.contains("config") && j["config"].contains("submodels"))
  {
    for (const auto& sm : j["config"]["submodels"])
      if (sm.contains("model"))
        n += weight_total(sm["model"]);
  }
  return n;
}

void measure(const std::string& path, Arena& arena)
{
  const std::vector<char> text = load(path);

  const size_t slash = path.find_last_of("/\\");
  std::printf("\n  %s  (%.1f KiB of JSON)\n", path.substr(slash + 1).c_str(), kib(text.size()));

  // --- stock nlohmann, for the equivalence check and a heap baseline --------
  g_heap_live = g_heap_peak = g_heap_total = g_heap_count = 0;
  g_counting = true;
  size_t stock_weights = 0;
  size_t stock_peak = 0;
  {
    nlohmann::json ref = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
    check(!ref.is_discarded(), "stock nlohmann parses it");
    stock_weights = weight_total(ref);
    stock_peak = g_heap_peak;
  }
  g_counting = false;

  // --- through the arena ----------------------------------------------------
  arena.Reset();
  arena.ResetStatistics();
  g_heap_live = g_heap_peak = g_heap_total = g_heap_count = 0;

  size_t arena_high = 0;
  size_t arena_heap_peak = 0;
  size_t arena_heap_count = 0;
  bool exhausted = true;
  size_t arena_weights = 0;
  {
    ScopedBind bind(arena);
    g_counting = true;
    arena_json doc = arena_json::parse(text.begin(), text.end(), nullptr, false);
    check(!doc.is_discarded(), "arena_json parses it");

    arena_weights = weight_total(doc);
    exhausted = arena.Exhausted();
    arena_high = arena.HighWater();
    arena_heap_peak = g_heap_peak;
    arena_heap_count = g_heap_count;

    // Spot-check that the arena DOM carries the same values, not just the same
    // shape -- an allocator bug would show up as corrupted floats, not as a
    // parse failure.
    if (doc.contains("architecture"))
      check(doc["architecture"].get<std::string>() == "SlimmableContainer"
              || doc["architecture"].get<std::string>() == "WaveNet",
            "architecture round-trips");
    g_counting = false;
  }

  check(!exhausted, "arena was large enough (no spill to heap)");
  check(arena_weights == stock_weights, "same weight count as stock nlohmann");

  std::printf("      arena high-water     %8.1f KiB   (%.1fx the JSON text)\n", kib(arena_high),
              static_cast<double>(arena_high) / static_cast<double>(text.size()));
  std::printf("      still on the heap    %8.1f KiB   in %zu allocations (strings)\n", kib(arena_heap_peak),
              arena_heap_count);
  std::printf("      stock nlohmann heap  %8.1f KiB   (all of it, for comparison)\n", kib(stock_peak));
  std::printf("      weights parsed       %8zu\n", arena_weights);
}

} // namespace

#ifndef NAMB_TEST_DATA
  #define NAMB_TEST_DATA "."
#endif

int main(int argc, char** argv)
{
  // With no arguments, the checked-in fixtures. Their weight arrays are
  // truncated, so the high-water figures they print are a fraction of what a
  // real model costs -- pass namb/active_models/*.nam to measure that.
  std::vector<std::string> models;
  if (argc == 1)
  {
    const std::string data = NAMB_TEST_DATA;
    models = {data + "/fixtures/a2_container.nam", data + "/fixtures/a2_lite.nam"};
  }
  else
  {
    for (int i = 1; i < argc; i++)
      models.emplace_back(argv[i]);
  }

  // Deliberately generous: the point of this run is to find the high-water
  // mark, not to prove a guess about it.
  static std::vector<uint8_t> store(8u * 1024u * 1024u);
  Arena arena(store.data(), store.size());

  std::printf("arena behaviour\n");
  {
    // Exhaustion must be survivable and visible, since that is how an
    // undersized arena will present on the target.
    static uint8_t tiny_store[256];
    Arena tiny(tiny_store, sizeof(tiny_store));
    check(tiny.Allocate(64, 8) != nullptr, "small allocation succeeds");
    check(tiny.Allocate(1024, 8) == nullptr, "oversized allocation returns null");
    check(tiny.Exhausted(), "and latches Exhausted()");
    tiny.Reset();
    check(!tiny.Exhausted(), "Reset clears the latch");
    check(tiny.Used() == 0, "Reset returns every byte");
    check(tiny.HighWater() >= 64, "high-water survives Reset");

    void* a = tiny.Allocate(1, 1);
    void* b = tiny.Allocate(1, 64);
    check((reinterpret_cast<uintptr_t>(b) % 64) == 0, "alignment is honoured");
    check(tiny.Owns(a) && tiny.Owns(b), "Owns() recognises its own blocks");
    check(!tiny.Owns(store.data()), "Owns() rejects a foreign pointer");

    Arena null_arena(nullptr, 0);
    check(null_arena.Allocate(8, 8) == nullptr, "unbacked arena allocates nothing");
    check(!null_arena.Owns(a), "unbacked arena owns nothing");
  }

  {
    // With nothing bound, the allocator must still work -- host tools use the
    // same type without an arena.
    nam::json_arena::ArenaAllocator<int> alloc;
    int* p = alloc.allocate(4);
    check(p != nullptr, "allocator falls back to the heap when unbound");
    p[0] = 1;
    p[3] = 4;
    check(p[0] == 1 && p[3] == 4, "the fallback block is usable");
    alloc.deallocate(p, 4);
  }

  std::printf("\nmodels\n");
  for (const std::string& m : models)
    measure(m, arena);

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
