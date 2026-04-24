//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// Runtime SIMD tier dispatcher: detects CPU features once and binds a table of
// function pointers to either the scalar or AVX-512 implementation of each
// kernel defined in simd_kernels.h.
//
// Usage:
//   world::simd::Initialize();           // safe to call multiple times
//   world::simd::SpectrumMultiplyHalf(...)  // dispatched call
//
// For benchmarking, world::simd::ForceTier() can override the detected tier
// (if the CPU doesn't support the requested tier, falls back to Scalar).
//-----------------------------------------------------------------------------
#ifndef WORLD_SIMD_DISPATCH_H_
#define WORLD_SIMD_DISPATCH_H_

#include "simd/simd_kernels.h"

namespace world {
namespace simd {

enum class Tier {
  Scalar = 0,
  AVX512 = 1,
};

// Thread-safe. First call detects CPU, binds function pointers, and reads the
// WORLD_FORCE_SIMD_TIER environment variable (accepted values: "scalar",
// "avx512"). Subsequent calls are no-ops unless ForceTier() was called.
void Initialize();

// Returns the currently active tier.
Tier GetActiveTier();

// Human-readable name: "Scalar" or "AVX512".
const char *GetActiveTierName();

// Overrides the detected tier. If the requested tier is unavailable on this
// CPU (or not compiled in), falls back to Scalar. Implicitly calls
// Initialize() if needed. Rebinds the function pointers.
void ForceTier(Tier tier);

// Returns true if the CPU supports AVX-512F + DQ + BW + VL AND the build has
// WORLD_HAS_AVX512 defined. Detection is lazy; calling this triggers
// Initialize() if needed.
bool CpuHasAvx512();

// Returns a string like "K1,K3" listing AVX-512 kernels disabled via the
// WORLD_AVX512_DISABLE env var. Returns an empty string if none disabled or
// if the Scalar tier is active. Implicitly calls Initialize().
// The returned buffer is static / thread-local; caller should not free.
const char *GetDisabledKernels();

// Dispatched function pointers. Initialize() binds these.
extern SpectrumMultiplyHalfFn SpectrumMultiplyHalf;
extern DcRemoveFn DcRemove;
extern CompositeF0MergeFn CompositeF0Merge;
extern ZcrSignChangeFn ZcrSignChange;
extern ZcrFrameSumFn ZcrFrameSum;
extern ZeroCrossingDetectAndCollectFn ZeroCrossingDetectAndCollect;
extern ZeroCrossingIntervalsFn ZeroCrossingIntervals;
extern GaussianFilter1DInteriorFn GaussianFilter1DInterior;
extern FastNuttallWindow8Fn FastNuttallWindow8;
extern CosineModulateInPlaceFn CosineModulateInPlace;

}  // namespace simd
}  // namespace world

#endif  // WORLD_SIMD_DISPATCH_H_
