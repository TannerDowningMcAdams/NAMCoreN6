// split_slimmable — extract the WaveNet submodels out of a SlimmableContainer .nam
//
// Every A2 model ships as architecture "SlimmableContainer" wrapping N complete
// sub-models (A2-Lite channels=3, A2-Full channels=8). Neither .namb v1 nor v2
// has a container architecture ID, so nam2namb cannot convert a container file
// at all. This tool lifts each submodel's "model" object -- itself a complete,
// self-contained .nam document with its own version/metadata/config/weights --
// into a standalone file that nam2namb and the A2 fast path both understand.
//
// Usage: split_slimmable <container.nam> <out_dir> [basename]
// Writes <out_dir>/<basename>_sub<i>_ch<C>.nam for each submodel.

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>

#include "json.hpp"

namespace fs = std::filesystem;
using nlohmann::json;

namespace
{
// Channel count of the (single) layer array, used only to name the output file
// so A2-Lite and A2-Full are distinguishable at a glance.
int layer_array_channels(const json& model)
{
  auto cfg = model.find("config");
  if (cfg == model.end())
    return -1;
  auto layers = cfg->find("layers");
  if (layers == cfg->end() || !layers->is_array() || layers->empty())
    return -1;
  const auto& l0 = (*layers)[0];
  auto ch = l0.find("channels");
  if (ch == l0.end() || !ch->is_number_integer())
    return -1;
  return ch->get<int>();
}
} // namespace

int main(int argc, char* argv[])
{
  if (argc < 3 || argc > 4)
  {
    std::cerr << "Usage: split_slimmable <container.nam> <out_dir> [basename]\n";
    return 2;
  }

  const fs::path inPath(argv[1]);
  const fs::path outDir(argv[2]);
  const std::string base = (argc >= 4) ? argv[3] : inPath.stem().string();

  std::ifstream in(inPath);
  if (!in.is_open())
  {
    std::cerr << "Error: cannot open " << inPath << "\n";
    return 1;
  }

  json doc;
  try
  {
    in >> doc;
  }
  catch (const std::exception& e)
  {
    std::cerr << "Error: " << inPath << " is not valid JSON: " << e.what() << "\n";
    return 1;
  }

  const std::string arch = doc.value("architecture", "");
  if (arch != "SlimmableContainer")
  {
    std::cerr << "Error: " << inPath << " has architecture \"" << arch
              << "\", expected \"SlimmableContainer\". Nothing to split -- feed this file "
                 "straight to nam2namb.\n";
    return 1;
  }

  auto cfg = doc.find("config");
  if (cfg == doc.end())
  {
    std::cerr << "Error: container has no \"config\" object\n";
    return 1;
  }
  auto submodels = cfg->find("submodels");
  if (submodels == cfg->end() || !submodels->is_array() || submodels->empty())
  {
    std::cerr << "Error: container has no non-empty \"submodels\" array\n";
    return 1;
  }

  std::error_code ec;
  fs::create_directories(outDir, ec);
  if (ec)
  {
    std::cerr << "Error: cannot create " << outDir << ": " << ec.message() << "\n";
    return 1;
  }

  // Container-level metadata (name, gear, input/output level) lives on the outer
  // document; the submodels carry only date/loudness/gain. Carry the outer keys
  // down so a split file reports the same sample rate and levels as the original.
  const json outerMeta = doc.value("metadata", json::object());

  int written = 0;
  for (size_t i = 0; i < submodels->size(); ++i)
  {
    const json& sm = (*submodels)[i];
    auto modelIt = sm.find("model");
    if (modelIt == sm.end() || !modelIt->is_object())
    {
      std::cerr << "Error: submodel " << i << " has no \"model\" object\n";
      return 1;
    }

    json model = *modelIt; // deep copy; we mutate metadata below

    if (outerMeta.is_object())
    {
      json merged = outerMeta;
      // Submodel's own metadata wins on conflict -- its loudness/gain are the
      // ones that belong to these weights. Bind the submodel metadata to a named
      // object first: value() returns a temporary, so begin()/end() taken from
      // two separate calls would be iterators into two different objects.
      const json innerMeta = model.value("metadata", json::object());
      for (auto it = innerMeta.begin(); it != innerMeta.end(); ++it)
        merged[it.key()] = it.value();
      merged["slimmable_source"] = inPath.filename().string();
      merged["slimmable_submodel_index"] = i;
      merged["slimmable_max_value"] = sm.value("max_value", 1.0);
      model["metadata"] = merged;
    }

    const int ch = layer_array_channels(model);
    std::string name = base + "_sub" + std::to_string(i);
    if (ch > 0)
      name += "_ch" + std::to_string(ch);
    name += ".nam";

    const fs::path outPath = outDir / name;
    std::ofstream out(outPath);
    if (!out.is_open())
    {
      std::cerr << "Error: cannot write " << outPath << "\n";
      return 1;
    }
    out << model.dump();
    if (!out.good())
    {
      std::cerr << "Error: write failed for " << outPath << "\n";
      return 1;
    }
    out.close();

    const size_t nWeights = model.value("weights", json::array()).size();
    std::cout << "wrote " << outPath.string() << "  (architecture=" << model.value("architecture", "?")
              << ", channels=" << ch << ", max_value=" << sm.value("max_value", 1.0) << ", weights=" << nWeights
              << ")\n";
    ++written;
  }

  std::cout << "split " << written << " submodel(s) from " << inPath.filename().string() << "\n";
  return 0;
}
