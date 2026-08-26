#pragma once
// Binary .namb loader for NAM models
// No dependency on nlohmann/json - suitable for embedded targets

#include <cstdint>
#include <memory>

#include <NAM/dsp.h>
#include <NAM/status.h>

#if !defined(NAM_NO_EXCEPTIONS)
  #include <filesystem>
#endif

namespace nam
{

/// \brief Load a NAM model from a .namb buffer, reporting failure by status.
///
/// The primary entry point, and the only one available when the library is
/// built without exceptions. Never throws and never allocates before the
/// container has been validated, so a corrupt or truncated blob is rejected
/// rather than partially parsed.
///
/// \param data   Pointer to the binary data. May point straight into
///               memory-mapped flash; the buffer is only read.
/// \param size   Bytes available at \p data.
/// \param status Set to Status::Ok on success, or the reason for failure.
/// \return The model, or nullptr on failure (in which case \p status says why).
std::unique_ptr<DSP> get_dsp_namb(const uint8_t* data, size_t size, Status& status);

#if !defined(NAM_NO_EXCEPTIONS)

/// \brief Load a NAM model from a .namb buffer.
/// \throws std::runtime_error on any failure, with the status text as the
///         message. Convenience wrapper over the Status overload, for callers
///         that already handle exceptions.
std::unique_ptr<DSP> get_dsp_namb(const uint8_t* data, size_t size);

/// \brief Load a NAM model from a .namb file.
///
/// Not available when NAM_NO_EXCEPTIONS is defined: it exists to read a file
/// and report failure by throwing, and an embedded target has neither
/// std::filesystem nor a reason to go through one -- the buffer overload above
/// takes a pointer into mapped flash directly.
///
/// \throws std::runtime_error on any failure.
std::unique_ptr<DSP> get_dsp_namb(const std::filesystem::path& filename);

#endif // !NAM_NO_EXCEPTIONS

} // namespace nam
