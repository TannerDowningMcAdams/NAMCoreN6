#pragma once

// Result codes for the loader / model-construction path.
//
// The library reports failure by throwing. On a bare-metal target the firmware
// is built -fno-exceptions, where a `throw` is a hard compile error and the
// unwinder is dead weight, so the paths that target needs -- the .namb loader
// and the A2 fast path -- also report through a Status instead.
//
// The two styles coexist. NAM_NO_EXCEPTIONS selects which one a translation
// unit gets: with it defined, the converted paths return Status and never
// throw; without it, behaviour is unchanged and existing desktop callers that
// catch std::runtime_error keep working.
//
// Style follows the firmware's device drivers (Ak4452, Ak5552): a small
// enum class : uint8_t, Ok == 0, and the object holds its last status so a
// caller can interrogate it after the fact rather than checking every call.

#include <cstdint>

namespace nam
{

/// \brief Outcome of a load or construction step. Ok is always zero, so
///        `if (status != Status::Ok)` reads the same everywhere.
enum class Status : uint8_t
{
  Ok = 0,

  // --- Container format (.namb) ---
  ErrorTooSmall,          ///< Buffer smaller than the fixed header + metadata block
  ErrorBadMagic,          ///< Magic number is not "NAMB"
  ErrorUnsupportedVersion,///< Format version outside [MIN_FORMAT_VERSION, FORMAT_VERSION]
  ErrorTruncated,         ///< A read ran past the end of the buffer
  ErrorChecksum,          ///< CRC32 mismatch
  ErrorWeightsOutOfRange, ///< Weight section extends beyond the declared file size

  // --- Model description ---
  ErrorUnknownArchitecture, ///< No parser registered for this architecture ID
  ErrorUnsupportedModelVersion, ///< Model config version outside the supported range
  ErrorInvalidConfig,     ///< Structurally valid but internally inconsistent
  ErrorUnsupportedShape,  ///< A shape this build has no implementation for

  // --- Weights ---
  ErrorWeightCount,       ///< Weight stream too short, or unread weights left over

  Error                   ///< Unclassified failure
};

/// \brief Convenience predicate, so call sites read as intent rather than as a
///        comparison against zero.
inline bool IsOk(Status s)
{
  return s == Status::Ok;
}

/// \brief Short, static description. Never allocates, safe to call from any
///        context, and returns a pointer with static storage duration.
const char* ToString(Status status);

// ============================================================================
// Failure latch
//
// Most of the library's validation lives in constructors, which have no return
// value to carry a status. Rather than restructure every one into a two-phase
// init, a failure records itself here and the loader checks once after
// construction -- so a fault still propagates to the caller, just not through
// the signatures.
//
// First failure wins: later calls do not overwrite an already-latched status,
// so the reported cause is the root one rather than whatever failed last as a
// consequence. Clear before a load, check after.
//
// Not thread-safe by design. Model loading happens on one thread; making this
// atomic would cost every embedded target for a case that does not arise.
// ============================================================================

/// \brief Latch a failure, if none is latched already.
void SetLastError(Status status);

/// \brief The latched failure, or Status::Ok if none.
Status GetLastError();

/// \brief Reset the latch. Call before a load.
void ClearLastError();

} // namespace nam

// ----------------------------------------------------------------------------
// Failure macros
//
// NAM_FAIL       - latch and carry on (constructors, void functions)
// NAM_FAIL_RET   - latch and return a value
//
// With exceptions enabled both throw, so desktop behaviour and existing catch
// sites are unchanged. Without them, the failure is latched and the caller
// finds it via GetLastError(). `msg` is used only by the throwing form; it is
// deliberately not stored in the no-exceptions build, which would mean
// formatting and allocating a string on a failure path.
// ----------------------------------------------------------------------------

#if defined(NAM_NO_EXCEPTIONS)

  #define NAM_FAIL(status, msg) ::nam::SetLastError(status)

  #define NAM_FAIL_RET(status, msg, ret)                                                                               \
    do                                                                                                                 \
    {                                                                                                                  \
      ::nam::SetLastError(status);                                                                                     \
      return ret;                                                                                                      \
    } while (0)

#else

  #include <stdexcept>
  #include <string>

  #define NAM_FAIL(status, msg) throw std::runtime_error(msg)
  #define NAM_FAIL_RET(status, msg, ret) throw std::runtime_error(msg)

#endif
