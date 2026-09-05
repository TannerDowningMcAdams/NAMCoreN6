# Host tools for the .namb / .nambpack pipeline.
#
#   .nam  --nam2namb-->  .namb  --nambpack-->  modelpack.bin  --> flash
#
# Deliberately a standalone project rather than part of the firmware build:
# Appli is cross-compiled for cortex-m55, and a tool built there would be an ARM
# binary you cannot run. Configure this with your host compiler instead.
#
# From the firmware repo root:
#   cmake -S NAMCoreN6/namb/tools -B build/tools
#   cmake --build build/tools
#
# Binaries land in build/tools/ (or build/tools/Release/ with MSVC).

cmake_minimum_required(VERSION 3.15)
project(namb_tools CXX)

# A cross toolchain leaking in here is the mistake this file exists to prevent,
# so fail loudly rather than silently emitting an unrunnable binary.
if(CMAKE_CROSSCOMPILING)
    message(FATAL_ERROR
        "namb_tools must be built for the host, but CMAKE_CROSSCOMPILING is set "
        "(toolchain: ${CMAKE_CXX_COMPILER}).\n"
        "Configure without the firmware toolchain file, e.g.:\n"
        "  cmake -S NAMCoreN6/namb/tools -B build/tools")
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Pin both binaries to <build>/bin regardless of generator.
#
# Multi-config generators (Visual Studio, Xcode) otherwise bury them in a
# per-config subdirectory -- build/tools/Release/nambpack.exe -- while Ninja and
# Makefiles put them at build/tools/nambpack. Since these get invoked by hand
# and by scripts, one predictable path is worth more than the config split;
# nothing here differs between Debug and Release anyway.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
foreach(_cfg ${CMAKE_CONFIGURATION_TYPES})
    string(TOUPPER "${_cfg}" _cfg_upper)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${_cfg_upper} "${CMAKE_BINARY_DIR}/bin")
endforeach()

# <core>/  -- the root that makes "NAM/..." and "namb/..." resolve.
cmake_path(SET NAM_CORE_ROOT "${CMAKE_CURRENT_LIST_DIR}/../..")
cmake_path(NORMAL_PATH NAM_CORE_ROOT)

foreach(_probe "${NAM_CORE_ROOT}/namb/nambpack_format.h" "${NAM_CORE_ROOT}/NAM/dsp.h")
    if(NOT EXISTS "${_probe}")
        message(FATAL_ERROR "Expected to find ${_probe} -- is NAMCoreN6 checked out?")
    endif()
endforeach()

set(NAM_DEPS "${NAM_CORE_ROOT}/Dependencies")

# --- nambpack ----------------------------------------------------------------
# Self-contained: it validates the .namb container and lays out bytes, but never
# parses a model, so it needs no part of the NAM library and no Eigen. That is
# why it is one add_executable with two include paths and nothing else.
add_executable(nambpack nambpack.cpp)
target_include_directories(nambpack PRIVATE "${NAM_CORE_ROOT}")

# --- nam2namb ----------------------------------------------------------------
# A shell over namb/namb_writer.h, which is also what the firmware compiles, so
# the tool that builds a model pack and the pedal that grows one cannot produce
# different bytes.
#
# It used to link the whole NAM library and Eigen, for one call into
# activations::ActivationConfig::from_json. The writer reimplements that mapping
# -- it has to, being generic over the JSON type -- so all that is needed now is
# status.cpp, for ToString().
add_executable(nam2namb nam2namb.cpp "${NAM_CORE_ROOT}/NAM/status.cpp")
target_include_directories(nam2namb PRIVATE
    "${NAM_CORE_ROOT}"
    "${NAM_DEPS}/nlohmann")

if(MSVC)
    target_compile_definitions(nam2namb PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
    target_compile_options(nam2namb PRIVATE /bigobj)  # nlohmann's DOM templates
else()
    target_compile_options(nam2namb PRIVATE -Wno-psabi)
endif()

# --- split_slimmable ---------------------------------------------------------
# Every A2 model ships as a SlimmableContainer wrapping A2-Lite and A2-Full, and
# .namb has no container architecture ID -- so this is the first step of the
# pipeline, not an optional extra. Needs only nlohmann: it lifts each submodel's
# "model" object (already a complete .nam document) out of the container without
# interpreting any of it.
add_executable(split_slimmable split_slimmable.cpp)
target_include_directories(split_slimmable PRIVATE "${NAM_DEPS}/nlohmann")

if(MSVC)
    target_compile_definitions(split_slimmable PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
endif()

add_custom_target(tools ALL DEPENDS nambpack nam2namb split_slimmable)

# --- tests -------------------------------------------------------------------
# Host tests for the parts the firmware shares: the pack format and its reader,
# the JSON arena, and the .nam -> .namb writer. They run without a target board
# and without the model library, on checked-in fixtures.
#
#   cmake --build <build> && ctest --test-dir <build> --output-on-failure

enable_testing()

set(NAMB_TESTS_DIR "${CMAKE_CURRENT_LIST_DIR}/../tests")

foreach(_test pack_test pack_writer_test arena_test writer_test)
    add_executable(${_test} "${NAMB_TESTS_DIR}/${_test}.cpp" "${NAM_CORE_ROOT}/NAM/status.cpp")
    target_include_directories(${_test} PRIVATE "${NAM_CORE_ROOT}" "${NAM_DEPS}/nlohmann")

    # Absolute, so a test can be run from any working directory -- ctest and a
    # hand-run from the bin/ directory both have to find the fixtures.
    target_compile_definitions(${_test} PRIVATE "NAMB_TEST_DATA=\"${NAMB_TESTS_DIR}\"")

    if(MSVC)
        target_compile_definitions(${_test} PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN)
        target_compile_options(${_test} PRIVATE /bigobj)
    else()
        target_compile_options(${_test} PRIVATE -Wno-psabi)
    endif()

    add_test(NAME ${_test} COMMAND ${_test})
endforeach()

add_custom_target(tests DEPENDS pack_test pack_writer_test arena_test writer_test)
