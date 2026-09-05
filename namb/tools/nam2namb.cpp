// nam2namb: convert a .nam (JSON) model to .namb (compact binary).
//
// Usage: nam2namb [options] input.nam [output.namb]
//
// Options:
//   -c, --channels <n>   Slimmable submodel to take, by channel count (default 3)
//   -n, --name <name>    Fallback entry name if the model carries none of its own
//   -h, --help
//
// Both a SlimmableContainer and a bare WaveNet are accepted. Containers used to
// require split_slimmable first; they no longer do, because the conversion has
// to run on the target as well and a pedal has nowhere to put an intermediate
// file. The split tool is still there, and build_modelpack.ps1 still uses it --
// its output converts to exactly the same bytes either way.
//
// This file is deliberately thin. Everything that decides what the bytes are
// lives in namb/namb_writer.h, which the firmware compiles too, so the tool
// that produces a model pack and the pedal that produces one cannot disagree.
// Only WaveNet and SlimmableContainer are converted; the .namb format can
// encode Linear, ConvNet and LSTM, but get_dsp_namb is built NAMB_WAVENET_ONLY
// and emitting a model the loader will refuse helps nobody.

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <json.hpp>
#include <namb/namb_writer.h>

namespace fs = std::filesystem;

namespace
{

void usage()
{
  std::fprintf(stderr,
               "Usage: nam2namb [options] input.nam [output.namb]\n"
               "\n"
               "  -c, --channels <n>   slimmable submodel to take (default 3)\n"
               "  -n, --name <name>    fallback entry name (default: the input's stem)\n"
               "\n"
               "Accepts a SlimmableContainer or a bare WaveNet model.\n");
}

bool read_file(const fs::path& p, std::vector<char>& out)
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
    f.read(out.data(), static_cast<std::streamsize>(out.size()));
  return f.good() || f.eof();
}

} // namespace

int main(int argc, char* argv[])
{
  fs::path input_path;
  fs::path output_path;
  nam::namb::WriteOptions opts;
  std::string fallback_name;

  for (int i = 1; i < argc; i++)
  {
    const std::string a = argv[i];

    if (a == "-h" || a == "--help")
    {
      usage();
      return 0;
    }
    if (a == "-c" || a == "--channels")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      opts.channels = static_cast<uint16_t>(std::stoul(argv[i]));
      continue;
    }
    if (a == "-n" || a == "--name")
    {
      if (++i >= argc)
      {
        usage();
        return 1;
      }
      fallback_name = argv[i];
      continue;
    }
    if (!a.empty() && a[0] == '-')
    {
      std::fprintf(stderr, "nam2namb: unknown option '%s'\n", a.c_str());
      usage();
      return 1;
    }

    if (input_path.empty())
      input_path = a;
    else if (output_path.empty())
      output_path = a;
    else
    {
      std::fprintf(stderr, "nam2namb: unexpected argument '%s'\n", a.c_str());
      return 1;
    }
  }

  if (input_path.empty())
  {
    usage();
    return 1;
  }
  if (output_path.empty())
  {
    output_path = input_path;
    output_path.replace_extension(".namb");
  }
  if (fallback_name.empty())
    fallback_name = input_path.stem().string();
  opts.fallback_name = fallback_name.c_str();

  // --- read and parse -------------------------------------------------------
  std::vector<char> text;
  if (!read_file(input_path, text))
  {
    std::fprintf(stderr, "nam2namb: cannot read %s\n", input_path.string().c_str());
    return 1;
  }

  // allow_exceptions = false, matching how the firmware parses: the writer is
  // shared, so the tool exercising a different error path than the pedal would
  // defeat the point of sharing it.
  nlohmann::json doc = nlohmann::json::parse(text.begin(), text.end(), nullptr, false);
  if (doc.is_discarded())
  {
    std::fprintf(stderr, "nam2namb: %s is not valid JSON\n", input_path.string().c_str());
    return 1;
  }

  // --- convert --------------------------------------------------------------
  // A float costs four bytes here and never fewer than six characters as JSON,
  // and the config block is small beside the weights, so the source length plus
  // a fixed margin is always enough.
  std::vector<uint8_t> blob(text.size() + 64 * 1024);

  nam::namb::WriteResult result;
  const nam::Status status = nam::namb::WriteNamb(doc, blob.data(), blob.size(), result, opts);
  if (!nam::IsOk(status))
  {
    std::fprintf(stderr, "nam2namb: %s\n", result.detail);
    std::fprintf(stderr, "          (%s)\n", nam::ToString(status));
    return 1;
  }

  // --- write ----------------------------------------------------------------
  std::ofstream out(output_path, std::ios::binary | std::ios::trunc);
  if (!out.is_open())
  {
    std::fprintf(stderr, "nam2namb: cannot create %s\n", output_path.string().c_str());
    return 1;
  }
  out.write(reinterpret_cast<const char*>(blob.data()), static_cast<std::streamsize>(result.size));
  if (!out.good())
  {
    std::fprintf(stderr, "nam2namb: write failed for %s\n", output_path.string().c_str());
    return 1;
  }
  out.close();

  // --- report ---------------------------------------------------------------
  const double reduction = 100.0 * (1.0 - static_cast<double>(result.size) / static_cast<double>(text.size()));
  std::printf("%s -> %s\n", input_path.filename().string().c_str(), output_path.filename().string().c_str());
  std::printf("  JSON: %zu bytes\n", text.size());
  std::printf("  NAMB: %zu bytes\n", result.size);
  std::printf("  Reduction: %.1f%%\n", reduction);
  std::printf("  %u weights, %u channels, entry name \"%s\"\n", result.weight_count,
              static_cast<unsigned>(result.channels), result.name);

  return 0;
}
