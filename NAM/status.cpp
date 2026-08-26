#include "status.h"

namespace nam
{

const char* ToString(Status status)
{
  switch (status)
  {
    case Status::Ok: return "Ok";
    case Status::ErrorTooSmall: return "buffer too small for header";
    case Status::ErrorBadMagic: return "invalid magic number";
    case Status::ErrorUnsupportedVersion: return "unsupported container format version";
    case Status::ErrorTruncated: return "unexpected end of data";
    case Status::ErrorChecksum: return "checksum mismatch";
    case Status::ErrorWeightsOutOfRange: return "weights extend beyond file";
    case Status::ErrorUnknownArchitecture: return "unknown architecture";
    case Status::ErrorUnsupportedModelVersion: return "unsupported model config version";
    case Status::ErrorInvalidConfig: return "invalid model configuration";
    case Status::ErrorUnsupportedShape: return "unsupported model shape";
    case Status::ErrorWeightCount: return "weight count mismatch";
    case Status::Error: return "error";
  }
  return "unknown status";
}

namespace
{
Status g_last_error = Status::Ok;
}

void SetLastError(Status status)
{
  // First failure wins, so the latched cause is the root one and not whatever
  // failed afterwards as a consequence of it.
  if (g_last_error == Status::Ok)
    g_last_error = status;
}

Status GetLastError()
{
  return g_last_error;
}

void ClearLastError()
{
  g_last_error = Status::Ok;
}

} // namespace nam
