// Golden-file test for namb_writer.
//
// tests/golden/*.namb were produced by the ORIGINAL three-stage host pipeline
// -- split_slimmable piped into the pre-rewrite nam2namb, the code that had
// been generating every model pack flashed to a pedal. They are checked in
// precisely because that pipeline no longer exists to ask: nam2namb is now a
// thin shell over namb_writer.h, so it can no longer serve as an independent
// oracle. These bytes are the last output of the code that came before, and
// the writer is required to keep reproducing them exactly.
//
// The fixtures are real A2 models -- the full config, dilations, kernel sizes,
// FiLM blocks and head verbatim -- with their weight arrays truncated to 32
// values so the pair is a few KB rather than 288. Both forms are covered:
//
//   a2_container.nam   a SlimmableContainer, exercising the submodel walk and
//                      the metadata fold (input_level_dbu and output_level_dbu
//                      exist only on the outer document)
//   a2_lite.nam        the bare ch3 submodel, which must NOT acquire those
//                      fields -- the two goldens differ in meta_flags and in
//                      the two level doubles, and that difference is the fold
//
// Each document is also converted twice, once through nlohmann::json and once
// through the arena DOM, and the results compared. That is what proves the
// writer's template is genuinely allocator-agnostic rather than accidentally
// working for the one instantiation it was developed against.
//
//   writer_test                            the checked-in fixtures
//   writer_test <model.nam> <expected.namb> [...]   a real model pack pair

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <namb/json_arena.h>
#include <namb/namb_writer.h>

using nam::json_arena::arena_json;
using nam::json_arena::Arena;
using nam::json_arena::ScopedBind;
using nam::namb::WriteNamb;
using nam::namb::WriteOptions;
using nam::namb::WriteResult;

namespace
{
int failures = 0;

void check(bool cond, const char* what)
{
  std::printf("      %-50s %s\n", what, cond ? "ok" : "FAIL");
  if (!cond)
    failures++;
}

std::vector<uint8_t> load(const std::string& path, bool required = true)
{
  std::ifstream f(path, std::ios::binary | std::ios::ate);
  if (!f.is_open())
  {
    if (required)
    {
      std::fprintf(stderr, "cannot open %s\n", path.c_str());
      std::exit(2);
    }
    return {};
  }
  const std::streamoff n = f.tellg();
  f.seekg(0);
  std::vector<uint8_t> v(static_cast<size_t>(n));
  f.read(reinterpret_cast<char*>(v.data()), n);
  return v;
}

// Where two blobs first differ, in terms a person can act on.
void report_diff(const std::vector<uint8_t>& a, const std::vector<uint8_t>& b)
{
  if (a.size() != b.size())
    std::printf("        sizes differ: %zu vs %zu\n", a.size(), b.size());

  const size_t n = a.size() < b.size() ? a.size() : b.size();
  for (size_t i = 0; i < n; i++)
  {
    if (a[i] != b[i])
    {
      const char* region = (i < 32) ? "file header" : (i < 80) ? "metadata block" : "model block or weights";
      std::printf("        first difference at byte %zu (%s): %02X vs %02X\n", i, region, a[i], b[i]);
      return;
    }
  }
}

const char* status_name(nam::Status s)
{
  return nam::ToString(s);
}

} // namespace

#ifndef NAMB_TEST_DATA
  #define NAMB_TEST_DATA "."
#endif

int main(int argc, char** argv)
{
  if (argc != 1 && ((argc - 1) % 2) != 0)
  {
    std::fprintf(stderr, "usage: writer_test [<model.nam> <expected.namb> ...]\n");
    return 2;
  }

  // With no arguments, the checked-in fixtures. NAMB_TEST_DATA is set by the
  // build so the test can be run from any working directory.
  std::vector<std::string> pairs;
  if (argc == 1)
  {
    const std::string data = NAMB_TEST_DATA;
    pairs = {
      data + "/fixtures/a2_container.nam", data + "/golden/a2_container_ch3.namb",
      data + "/fixtures/a2_lite.nam",      data + "/golden/a2_lite.namb",
    };
  }
  else
  {
    for (int i = 1; i < argc; i++)
      pairs.emplace_back(argv[i]);
  }

  static std::vector<uint8_t> arena_store(8u * 1024u * 1024u);
  Arena arena(arena_store.data(), arena_store.size());

  std::vector<uint8_t> blob(64 * 1024);

  for (size_t i = 0; i + 1 < pairs.size(); i += 2)
  {
    const std::string nam_path = pairs[i];
    const std::string namb_path = pairs[i + 1];

    const size_t slash = nam_path.find_last_of("/\\");
    const std::string file = nam_path.substr(slash + 1);
    std::printf("\n  %s\n", file.c_str());

    // What the target will pass: the SD filename, used only when the document
    // carries no name of its own.
    const std::string stem = file.substr(0, file.size() - 4);
    WriteOptions opts;
    opts.fallback_name = stem.c_str();

    const std::vector<uint8_t> text = load(nam_path);
    const std::vector<uint8_t> expected = load(namb_path);

    // --- through nlohmann::json ---------------------------------------------
    std::vector<uint8_t> via_stock;
    WriteResult r_stock;
    {
      nlohmann::json doc = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
      check(!doc.is_discarded(), "container parses");

      const nam::Status st = WriteNamb(doc, blob.data(), blob.size(), r_stock, opts);
      if (!nam::IsOk(st))
        std::printf("        %s: %s\n", status_name(st), r_stock.detail);
      check(nam::IsOk(st), "converts through nlohmann::json");
      via_stock.assign(blob.begin(), blob.begin() + r_stock.size);
    }

    // --- through the arena DOM ----------------------------------------------
    std::vector<uint8_t> via_arena;
    WriteResult r_arena;
    {
      arena.Reset();
      ScopedBind bind(arena);
      arena_json doc = arena_json::parse(text.begin(), text.end(), nullptr, false);
      check(!doc.is_discarded(), "container parses into the arena");

      const nam::Status st = WriteNamb(doc, blob.data(), blob.size(), r_arena, opts);
      if (!nam::IsOk(st))
        std::printf("        %s: %s\n", status_name(st), r_arena.detail);
      check(nam::IsOk(st), "converts through arena_json");
      check(!arena.Exhausted(), "arena was large enough");
      via_arena.assign(blob.begin(), blob.begin() + r_arena.size);
    }

    // --- the three-way comparison -------------------------------------------
    const bool same_dom = (via_stock == via_arena);
    check(same_dom, "both DOM types give identical bytes");
    if (!same_dom)
      report_diff(via_stock, via_arena);

    const bool matches_oracle = (via_stock == expected);
    check(matches_oracle, "identical to split_slimmable + nam2namb output");
    if (!matches_oracle)
      report_diff(via_stock, expected);

    check(r_stock.channels == 3, "reports the channels=3 submodel");
    // The exact count is already pinned by the golden comparison; what matters
    // here is that both DOM types walked the same weights.
    check(r_stock.weight_count > 0, "found weights");
    check(r_stock.weight_count == r_arena.weight_count, "same weight count from both DOM types");
    check(r_stock.name[0] != '\0', "derived a pack entry name");
    check(std::strlen(r_stock.name) < nam::namb::kNameSize, "name fits a pack entry");
    check(std::strcmp(r_stock.name, r_arena.name) == 0, "same name from both DOM types");
    check(std::strcmp(r_stock.name, "model") != 0, "named from metadata or the filename, never the sentinel");
    std::printf("        %zu bytes, %u weights, name \"%s\"\n", r_stock.size, r_stock.weight_count, r_stock.name);
  }

  // --- failure paths --------------------------------------------------------
  // The point of the rewrite was that a bad document reports rather than
  // aborts, so the malformed cases matter as much as the good ones.
  std::printf("\n  malformed input\n");
  {
    struct Case
    {
      const char* json;
      const char* what;
    };
    const Case cases[] = {
      {R"({"architecture":"LSTM","config":{},"weights":[]})", "unsupported architecture is refused"},
      {R"({"config":{}})", "missing architecture is refused"},
      {R"({"architecture":"WaveNet"})", "WaveNet with no config is refused"},
      {R"({"architecture":"SlimmableContainer","config":{"submodels":[]}})", "empty container is refused"},
      {R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[]}})", "empty layers is refused"},
      {R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
       R"("condition_size":1,"channels":3,"dilations":[1,2],"kernel_sizes":[3],)"
       R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]}})",
       "kernel_sizes/dilations mismatch is refused"},
      {R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
       R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
       R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Nonesuch"}]}})",
       "unknown activation is refused"},
      {R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
       R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
       R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]},)"
       R"("weights":["not a number"]})",
       "non-numeric weights are refused"},
    };

    for (const Case& c : cases)
    {
      nlohmann::json doc = nlohmann::json::parse(c.json, nullptr, false);
      WriteResult r;
      const nam::Status st = WriteNamb(doc, blob.data(), blob.size(), r);
      const bool refused = !nam::IsOk(st) && r.detail[0] != '\0';
      check(refused, c.what);
      if (!refused)
        std::printf("        expected a failure, got Ok\n");
    }

    // A container that holds only A2-Full must say so, and say what it had.
    {
      const char* json = R"({"architecture":"SlimmableContainer","config":{"submodels":[{"model":{)"
                         R"("architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"channels":8}]}}}]}})";
      nlohmann::json doc = nlohmann::json::parse(json, nullptr, false);
      WriteResult r;
      const nam::Status st = WriteNamb(doc, blob.data(), blob.size(), r);
      check(!nam::IsOk(st), "container without channels=3 is refused");
      check(std::strstr(r.detail, "8") != nullptr, "and the message names what it did hold");
      std::printf("        \"%s\"\n", r.detail);
    }

    // A short output span must be reported, not overrun.
    {
      const char* json = R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
                         R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
                         R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]},)"
                         R"("weights":[1.0,2.0]})";
      nlohmann::json doc = nlohmann::json::parse(json, nullptr, false);
      WriteResult r;
      uint8_t tiny[100];
      const nam::Status st = WriteNamb(doc, tiny, sizeof(tiny), r);
      check(st == nam::Status::ErrorTooSmall, "a short output span reports ErrorTooSmall");

      // ...and the same document fits when the span is adequate.
      const nam::Status ok = WriteNamb(doc, blob.data(), blob.size(), r);
      check(nam::IsOk(ok), "and succeeds when given room");
      check(r.weight_count == 2, "with the weights it was given");
    }
  }

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
