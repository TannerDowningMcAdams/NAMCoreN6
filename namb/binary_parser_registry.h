#pragma once
// Binary config parser registry for .namb format
// Mirrors ConfigParserRegistry from model_config.h but maps uint8_t architecture IDs
// to binary parser functions instead of string names to JSON parsers.
//
// Parsers receive the file's format_version so that they can read version-dependent
// fields. See namb_format.h for the version history; today only the WaveNet parser
// varies (v2 added per-layer kernel_sizes, head_kernel_size and head_dilation).

#include <cstdint>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <NAM/model_config.h>
#include <NAM/status.h>
#include "namb_format.h"

namespace nam
{
namespace namb
{

using BinaryConfigParserFunction = std::function<std::unique_ptr<ModelConfig>(
    BinaryReader& reader, const float*& weights, size_t& weight_count, const ModelMetadata& meta,
    uint16_t format_version, Status& status)>;

class BinaryConfigParserRegistry
{
public:
  static BinaryConfigParserRegistry& instance()
  {
    static BinaryConfigParserRegistry registry;
    return registry;
  }

  void registerParser(uint8_t arch_id, BinaryConfigParserFunction func)
  {
    parsers_[arch_id] = std::move(func);
  }

  bool has(uint8_t arch_id) const { return parsers_.find(arch_id) != parsers_.end(); }

  /// \brief Dispatch to the parser registered for arch_id.
  /// \param status Set to ErrorUnknownArchitecture when no parser is registered,
  ///               left untouched otherwise. Never throws, so the same registry
  ///               serves builds compiled without exceptions; the caller decides
  ///               whether to surface the failure as a Status or a throw.
  /// \return The parsed config, or nullptr on failure.
  std::unique_ptr<ModelConfig> parse(uint8_t arch_id, BinaryReader& reader, const float*& weights,
                                      size_t& weight_count, const ModelMetadata& meta,
                                      uint16_t format_version, Status& status) const
  {
    auto it = parsers_.find(arch_id);
    if (it == parsers_.end())
    {
      status = Status::ErrorUnknownArchitecture;
      return nullptr;
    }
    return it->second(reader, weights, weight_count, meta, format_version, status);
  }

private:
  BinaryConfigParserRegistry() = default;
  std::unordered_map<uint8_t, BinaryConfigParserFunction> parsers_;
};

struct BinaryConfigParserHelper
{
  BinaryConfigParserHelper(uint8_t arch_id, BinaryConfigParserFunction func)
  {
    BinaryConfigParserRegistry::instance().registerParser(arch_id, std::move(func));
  }
};

} // namespace namb
} // namespace nam
