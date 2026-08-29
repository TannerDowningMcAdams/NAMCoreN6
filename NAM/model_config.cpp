// Construction path shared by every loader.
//
// create_dsp() lives here rather than in get_dsp.cpp so an embedded target can
// compile it without dragging in that file's JSON readers, mutex-guarded
// version registry and <iostream>. A .namb loader needs these four lines alone.
//
// dsp.h is included explicitly: model_config.h only forward-declares DSP, which
// is enough to pass a unique_ptr<DSP> around but not to call members on one.

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
