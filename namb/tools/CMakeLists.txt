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
# Unlike nambpack, this one does parse models, so it needs the core and its
# dependencies. Kept here anyway so the whole pipeline is available from one
# build directory -- having to cross repos midway through converting a model is
# how mismatched tool versions creep in.
file(GLOB NAM_SOURCES
    "${NAM_CORE_ROOT}/NAM/*.cpp"
    "${NAM_CORE_ROOT}/NAM/*/*.cpp")

add_executable(nam2namb nam2namb.cpp ${NAM_SOURCES})
target_include_directories(nam2namb PRIVATE
    "${NAM_CORE_ROOT}"
    "${NAM_DEPS}/nlohmann")
target_include_directories(nam2namb SYSTEM PRIVATE "${NAM_DEPS}/eigen")
target_compile_definitions(nam2namb PRIVATE NAM_ENABLE_A2_FAST)
target_compile_features(nam2namb PRIVATE cxx_std_20)

if(MSVC)
    target_compile_definitions(nam2namb PRIVATE NOMINMAX WIN32_LEAN_AND_MEAN _USE_MATH_DEFINES)
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
