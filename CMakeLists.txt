# CMake build system design considerations:
# - Include directories:
#   + Do not define include directories globally using the include_directories
#     command but rather at the target level using the
#     target_include_directories command. That way, it is easier to guarantee
#     that targets are built using the proper list of include directories.
#   + Use the PUBLIC and PRIVATE keywords to specify the scope of include
#     directories. That way, a target linking to a library (using the
#     target_link_librairies command) inherits from the library PUBLIC include
#     directories and not from the PRIVATE ones.

cmake_minimum_required(VERSION 3.0...3.30)

# Export compile_commands.json for IDE/clangd include-path resolution.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Build Options
option(WORLD_BUILD_TESTS "Set to ON to build tests" OFF)
option(WORLD_BUILD_EXAMPLES "Set to ON to build examples" ON)
set(WORLD_FFT_BACKEND "auto" CACHE STRING "FFT backend: auto, fftw3, or ooura")
set_property(CACHE WORLD_FFT_BACKEND PROPERTY STRINGS auto fftw3 ooura)
set(WORLD_AOCL_FFTW_DIR "" CACHE PATH
    "Optional AOCL-FFTW install prefix (contains lib/fftw3.lib + include/fftw3.h). \
When set, this takes precedence over a generic system fftw3.")

# FFT backend resolution.
#
# Priority when WORLD_FFT_BACKEND=auto or fftw3:
#   1. WORLD_AOCL_FFTW_DIR (CMake cache) -> AOCL-FFTW
#   2. $ENV{AOCL_ROOT}                   -> AOCL-FFTW
#   3. find_library(fftw3) in system     -> upstream FFTW
#   4. (auto only) fall back to Ooura
set(WORLD_FFT_BACKEND_KIND "ooura")
if(WORLD_FFT_BACKEND STREQUAL "auto" OR WORLD_FFT_BACKEND STREQUAL "fftw3")
  set(_world_aocl_hint "")
  if(WORLD_AOCL_FFTW_DIR)
    set(_world_aocl_hint "${WORLD_AOCL_FFTW_DIR}")
  elseif(DEFINED ENV{AOCL_ROOT})
    set(_world_aocl_hint "$ENV{AOCL_ROOT}")
  endif()
  unset(FFTW3_LIB CACHE)
  if(_world_aocl_hint)
    find_library(FFTW3_LIB
        NAMES fftw3 libfftw3 libfftw3-3
        HINTS "${_world_aocl_hint}/lib" "${_world_aocl_hint}"
        NO_DEFAULT_PATH)
    if(FFTW3_LIB)
      set(WORLD_FFT_BACKEND_KIND "aocl")
      message(STATUS "FFT backend: AOCL-FFTW at ${FFTW3_LIB}")
    endif()
  endif()
  if(NOT FFTW3_LIB)
    find_library(FFTW3_LIB NAMES fftw3 libfftw3 libfftw3-3)
    if(FFTW3_LIB)
      set(WORLD_FFT_BACKEND_KIND "fftw3")
      message(STATUS "FFT backend: upstream FFTW3 at ${FFTW3_LIB}")
    endif()
  endif()
  if(FFTW3_LIB)
    set(WORLD_FFT_BACKEND "fftw3")
    # Also locate fftw3.h for include path (helpful when AOCL is non-system).
    if(_world_aocl_hint)
      find_path(FFTW3_INCLUDE_DIR fftw3.h
          HINTS "${_world_aocl_hint}/include" "${_world_aocl_hint}"
          NO_DEFAULT_PATH)
    endif()
    if(NOT FFTW3_INCLUDE_DIR)
      find_path(FFTW3_INCLUDE_DIR fftw3.h)
    endif()
  else()
    set(WORLD_FFT_BACKEND "ooura")
    set(WORLD_FFT_BACKEND_KIND "ooura")
    message(STATUS "FFT backend: Ooura (no fftw3 found)")
  endif()
endif()

# AVX-512 detection is deferred until after project() (needs a C++ compiler).

# use git version as library version
find_package(Git QUIET)
if (Git_FOUND AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
    execute_process(
            COMMAND ${GIT_EXECUTABLE} rev-list --count HEAD
            WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
            OUTPUT_VARIABLE git_version
            OUTPUT_STRIP_TRAILING_WHITESPACE
    )
else ()
    set(git_version 0)
endif ()

project(WORLD LANGUAGES CXX VERSION 1.0.${git_version})

# AVX-512 compiler-support detection. Per-file flags applied only to
# src/simd/simd_kernels_avx512.cpp so the rest of the library keeps its
# existing codegen (safe for CPUs without AVX-512 at runtime).
include(CheckCXXCompilerFlag)
if(MSVC)
  check_cxx_compiler_flag("/arch:AVX512" WORLD_COMPILER_HAS_AVX512)
  set(WORLD_AVX512_FLAGS "/arch:AVX512")
else()
  check_cxx_compiler_flag("-mavx512f" WORLD_COMPILER_HAS_AVX512)
  # Zen5: AVX-512F + DQ + BW + VL + FMA + BMI2.
  set(WORLD_AVX512_FLAGS
      "-mavx512f;-mavx512dq;-mavx512bw;-mavx512vl;-mfma;-mbmi2")
endif()

add_library(world STATIC
        src/world/cheaptrick.h
        src/world/codec.h
        src/world/common.h
        src/world/compositef0.h
        src/world/constantnumbers.h
        src/world/d4c.h
        src/world/dio.h
        src/world/fft.h
        src/world/harvest.h
        src/world/macrodefinitions.h
        src/world/matlabfunctions.h
        src/world/stonemask.h
        src/world/synthesis.h
        src/world/synthesisrealtime.h
        src/simd/simd_kernels.h
        src/simd/simd_dispatch.h
        src/simd/simd_kernels_scalar.cpp
        src/simd/simd_kernels_avx512.cpp
        src/simd/simd_dispatch.cpp
        src/cheaptrick.cpp
        src/codec.cpp
        src/common.cpp
        src/compositef0.cpp
        src/d4c.cpp
        src/dio.cpp
        $<IF:$<STREQUAL:${WORLD_FFT_BACKEND},fftw3>,src/fft_fftw3.cpp,src/fft.cpp>
        src/harvest.cpp
        src/matlabfunctions.cpp
        src/stonemask.cpp
        src/synthesis.cpp
        src/synthesisrealtime.cpp
        )

add_library(world_tool STATIC
        tools/audioio.h
        tools/parameterio.h
        tools/audioio.cpp
        tools/parameterio.cpp
        )

target_include_directories(world PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>)
target_include_directories(world_tool PUBLIC $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/tools>)

add_library(world::core ALIAS world)
add_library(world::tool ALIAS world_tool)

# Optimization flags for Release builds. MSVC does not understand -march=native;
# split the flag set per compiler id.
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
target_compile_options(world PRIVATE
  $<$<AND:$<CONFIG:Release>,$<NOT:$<CXX_COMPILER_ID:MSVC>>>:-O3;-march=native;-ffast-math;-funroll-loops>
  $<$<AND:$<CONFIG:Release>,$<CXX_COMPILER_ID:MSVC>>:/O2;/fp:fast>
)

# Per-file AVX-512 flags for the intrinsics kernel. When the compiler lacks
# AVX-512 support, the file still compiles but produces no avx512 symbols
# (WORLD_AVX512_STUB branch inside the .cpp).
if(WORLD_COMPILER_HAS_AVX512)
  set_source_files_properties(src/simd/simd_kernels_avx512.cpp
      PROPERTIES COMPILE_OPTIONS "${WORLD_AVX512_FLAGS}")
  target_compile_definitions(world PRIVATE WORLD_HAS_AVX512)
else()
  set_source_files_properties(src/simd/simd_kernels_avx512.cpp
      PROPERTIES COMPILE_DEFINITIONS "WORLD_AVX512_STUB")
endif()

# FFTW3 linking
if(WORLD_FFT_BACKEND STREQUAL "fftw3")
  target_link_libraries(world PUBLIC ${FFTW3_LIB})
  if(FFTW3_INCLUDE_DIR)
    target_include_directories(world PUBLIC $<BUILD_INTERFACE:${FFTW3_INCLUDE_DIR}>)
  endif()
endif()

foreach (lib world world_tool)
    set_target_properties(${lib}
            PROPERTIES
            ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib"
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin"
            )
endforeach ()

include(GNUInstallDirs)
install(TARGETS world world_tool
        EXPORT world-export
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        )
install(DIRECTORY src/world DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

include(CMakePackageConfigHelpers)
configure_package_config_file(world-config.cmake.in
        ${CMAKE_CURRENT_BINARY_DIR}/world-config.cmake
        INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/world/cmake
        PATH_VARS CMAKE_INSTALL_INCLUDEDIR
        )
write_basic_package_version_file(
        world-config-version.cmake
        VERSION ${PACKAGE_VERSION}
        COMPATIBILITY AnyNewerVersion
)
install(FILES ${CMAKE_CURRENT_BINARY_DIR}/world-config.cmake
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/world
        )
install(EXPORT world-export
        FILE world-config-version.cmake
        NAMESPACE world::
        DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/world
        )
export(TARGETS world world_tool NAMESPACE world::
        FILE ${PROJECT_BINARY_DIR}/world-targets.cmake)

if (WORLD_BUILD_TESTS)
    add_executable(tests test/test.cpp)
    target_link_libraries(tests
            PRIVATE world
            PRIVATE world_tool
            )
endif ()

if(WORLD_BUILD_EXAMPLES)
    add_subdirectory(examples)
endif()

