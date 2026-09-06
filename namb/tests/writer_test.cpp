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
// They have survived two rewrites unchanged -- the split of the byte layout
// into namb_config.h, and the replacement of the nlohmann DOM front end with a
// streaming parser -- which is the entire reason they exist.
//
// The fixtures are real A2 models -- the full config, dilations, kernel sizes,
// FiLM blocks and head verbatim -- with their weight arrays truncated to 32
// values so the pair is a few KB rather than 288. Both forms are covered:
//
//   a2_container.nam   a SlimmableContainer, exercising the submodel walk and
//                      the metadata fold (input_level_dbu and output_level_dbu
//                      exist only on the outer document). Converted at both
//                      channel counts: taking ch8 is the only thing that
//                      exercises rejecting a submodel and moving to the next
//   a2_lite.nam        the bare ch3 submodel, which must NOT acquire those
//                      fields -- the two goldens differ in meta_flags and in
//                      the two level doubles, and that difference is the fold
//
//   writer_test                            the checked-in fixtures
//   writer_test <model.nam> <expected.namb> [...]   a real model pack pair

#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

#include <namb/namb_writer.h>

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

  // Up to a few differences rather than only the first: a checksum mismatch is
  // a consequence of some earlier byte, and seeing one without the other says
  // nothing about which.
  const size_t n = a.size() < b.size() ? a.size() : b.size();
  size_t shown = 0;
  size_t total = 0;
  for (size_t i = 0; i < n; i++)
  {
    if (a[i] == b[i])
      continue;
    total++;
    if (shown < 12)
    {
      const char* region = (i < 32) ? "file header" : (i < 80) ? "metadata block" : "model block or weights";
      std::printf("        byte %zu (%s): %02X vs %02X\n", i, region, a[i], b[i]);
      shown++;
    }
  }
  if (total > shown)
    std::printf("        ... and %zu more differing bytes\n", total - shown);
}

const char* status_name(nam::Status s)
{
  return nam::ToString(s);
}

// The streaming front end's source contract: get() hands back the next byte, or
// something negative at the end. On the target this is backed by SdFs; here the
// document is already in memory.
struct MemSource
{
  const uint8_t* data;
  size_t size;
  size_t pos = 0;

  int get() { return (pos < size) ? static_cast<int>(data[pos++]) : -1; }
};

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

  // ~16 KB, and reused across every model below -- which is also the contract
  // the firmware holds it under.
  static nam::namb::StreamConverter stream_conv;

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

    // --- convert -------------------------------------------------------------
    std::vector<uint8_t> via_stream;
    WriteResult r;
    {
      MemSource src{text.data(), text.size()};
      const nam::Status st = nam::namb::WriteNambStream(src, blob.data(), blob.size(), r, opts, stream_conv);
      if (!nam::IsOk(st))
        std::printf("        %s: %s\n", status_name(st), r.detail);
      check(nam::IsOk(st), "converts");
      via_stream.assign(blob.begin(), blob.begin() + r.size);

      // Reading stops as soon as there is nothing left to learn, which for a
      // container whose metadata precedes its submodels is well before the end.
      std::printf("        streamed %zu of %zu source bytes\n", src.pos, text.size());
    }

    const bool matches_oracle = (via_stream == expected);
    check(matches_oracle, "identical to split_slimmable + nam2namb output");
    if (!matches_oracle)
      report_diff(expected, via_stream);

    check(r.channels == 3, "reports the channels=3 submodel");
    check(r.weight_count > 0, "found weights");

    // --- the other submodel, which the parser has to skip past ---------------
    // channels=8 is not what the pedal loads, but taking it is the only thing
    // that exercises rejecting a submodel: retiring the anchors, discarding an
    // already-parsed layer array, and picking up the next "model" object.
    const std::string ch8_golden = namb_path.substr(0, namb_path.size() - 5) + "8.namb";
    const std::vector<uint8_t> expected8 = load(ch8_golden, false);
    if (!expected8.empty())
    {
      WriteOptions opts8 = opts;
      opts8.channels = 8;

      MemSource src8{text.data(), text.size()};
      WriteResult r8;
      const nam::Status st8 = nam::namb::WriteNambStream(src8, blob.data(), blob.size(), r8, opts8, stream_conv);
      if (!nam::IsOk(st8))
        std::printf("        %s: %s\n", status_name(st8), r8.detail);
      check(nam::IsOk(st8), "converts the channels=8 submodel");

      const std::vector<uint8_t> via_stream8(blob.begin(), blob.begin() + r8.size);
      const bool same8 = (via_stream8 == expected8);
      check(same8, "skipping a submodel reproduces its golden too");
      if (!same8)
        report_diff(expected8, via_stream8);
      check(r8.channels == 8, "reports the channels=8 submodel");
    }

    check(r.name[0] != '\0', "derived a pack entry name");
    check(std::strlen(r.name) < nam::namb::kNameSize, "name fits a pack entry");
    check(std::strcmp(r.name, "model") != 0, "named from metadata or the filename, never the sentinel");
    std::printf("        %zu bytes, %u weights, name \"%s\"\n", r.size, r.weight_count, r.name);
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

    // A convenience for the cases below, all of which are short literals.
    auto convert = [&](const char* json, uint8_t* dst, size_t cap, WriteResult& r, uint16_t channels = 3) {
      MemSource src{reinterpret_cast<const uint8_t*>(json), std::strlen(json)};
      WriteOptions o;
      o.channels = channels;
      return nam::namb::WriteNambStream(src, dst, cap, r, o, stream_conv);
    };

    for (const Case& c : cases)
    {
      WriteResult r;
      const nam::Status st = convert(c.json, blob.data(), blob.size(), r);
      const bool refused = !nam::IsOk(st) && r.detail[0] != '\0';
      check(refused, c.what);
      if (!refused)
        std::printf("        expected a failure, got Ok\n");
    }

    // A container that holds only A2-Full must say so, and say what it had.
    {
      const char* json = R"({"architecture":"SlimmableContainer","config":{"submodels":[{"model":{)"
                         R"("architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"channels":8}]}}}]}})";
      WriteResult r;
      const nam::Status st = convert(json, blob.data(), blob.size(), r);
      check(!nam::IsOk(st), "container without channels=3 is refused");
      check(std::strstr(r.detail, "8") != nullptr, "and the message names what it did hold");
      std::printf("        \"%s\"\n", r.detail);
    }

    // A condition_dsp is refused deliberately rather than converted wrongly:
    // its weights precede the model block that would have to be sized before
    // they could be placed. No model has ever carried one.
    {
      const char* json = R"({"architecture":"WaveNet","version":"1.0.0","config":{"condition_dsp":{)"
                         R"("architecture":"WaveNet","weights":[1.0]},"layers":[{"input_size":1,)"
                         R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
                         R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]},)"
                         R"("weights":[1.0]})";
      WriteResult r;
      const nam::Status st = convert(json, blob.data(), blob.size(), r);
      check(!nam::IsOk(st), "a condition_dsp is refused, not mis-converted");
      check(std::strstr(r.detail, "condition_dsp") != nullptr, "and the message names it");
      std::printf("        \"%s\"\n", r.detail);
    }

    // A UTF-8 BOM is not whitespace to lwjson and would otherwise be refused on
    // the first byte. Half the models on hand carry one.
    {
      const char* json = "\xEF\xBB\xBF"
                         R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
                         R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
                         R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]},)"
                         R"("weights":[1.0,2.0]})";
      WriteResult r;
      const nam::Status st = convert(json, blob.data(), blob.size(), r);
      check(nam::IsOk(st), "a leading UTF-8 BOM is skipped");
      check(r.weight_count == 2, "and the model behind it converts");
    }

    // A short output span must be reported, not overrun.
    {
      const char* json = R"({"architecture":"WaveNet","version":"1.0.0","config":{"layers":[{"input_size":1,)"
                         R"("condition_size":1,"channels":3,"dilations":[1],"kernel_sizes":[3],)"
                         R"("head":{"out_channels":1,"kernel_size":1,"bias":true},"activation":"Tanh"}]},)"
                         R"("weights":[1.0,2.0]})";
      WriteResult r;
      uint8_t tiny[100];
      const nam::Status st = convert(json, tiny, sizeof(tiny), r);
      check(st == nam::Status::ErrorTooSmall, "a short output span reports ErrorTooSmall");

      // ...and the same document fits when the span is adequate.
      const nam::Status ok = convert(json, blob.data(), blob.size(), r);
      check(nam::IsOk(ok), "and succeeds when given room");
      check(r.weight_count == 2, "with the weights it was given");
    }
  }

  std::printf("\n%s (%d failure(s))\n", failures ? "FAILED" : "PASSED", failures);
  return failures ? 1 : 0;
}
