// Construction path shared by every loader.
//
// create_dsp() is declared in model_config.h but used to live in get_dsp.cpp,
// which also carries the JSON file readers, a mutex-guarded version-support
// registry and <iostream>. A binary loader (.namb) needs these four lines and
// none of that, so they live here instead -- an embedded target can compile
// this translation unit without pulling the JSON path in behind it.
//
// dsp.h is included explicitly: model_config.h only forward-declares DSP (to
// stay free of a circular include), which is enough to pass a unique_ptr<DSP>
// around but not to call member functions on one.

#include "model_config.h"

#include "dsp.h"

namespace nam
{

namespace
{

// Moved here with create_dsp, its only caller.
void apply_metadata(DSP& dsp, const ModelMetadata& metadata)
{
  if (metadata.loudness.has_value())
    dsp.SetLoudness(metadata.loudness.value());
  if (metadata.input_level.has_value())
    dsp.SetInputLevel(metadata.input_level.value());
  if (metadata.output_level.has_value())
    dsp.SetOutputLevel(metadata.output_level.value());
}

} // anonymous namespace

std::unique_ptr<DSP> create_dsp(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                const ModelMetadata& metadata)
{
  Status status = Status::Ok;
  return create_dsp(std::move(config), std::move(weights), metadata, status);
}

std::unique_ptr<DSP> create_dsp(std::unique_ptr<ModelConfig> config, std::vector<float> weights,
                                const ModelMetadata& metadata, Status& status)
{
  status = Status::Ok;

  if (config == nullptr)
  {
    status = Status::ErrorInvalidConfig;
    return nullptr;
  }

  auto out = config->create(std::move(weights), metadata.sample_rate);
  if (out == nullptr)
  {
    // Either create() reported failure directly, or something deeper latched it.
    status = IsOk(GetLastError()) ? Status::Error : GetLastError();
    return nullptr;
  }

  apply_metadata(*out, metadata);
  return out;
}

} // namespace nam
