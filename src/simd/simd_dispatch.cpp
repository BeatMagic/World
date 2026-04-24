//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// CPU feature detection + SIMD tier binding. Binds one function pointer per
// kernel. Thread-safe initialization via std::call_once.
//-----------------------------------------------------------------------------
#include "simd/simd_dispatch.h"

#include <stdlib.h>
#include <string.h>

#include <mutex>

#if defined(_MSC_VER)
#  include <intrin.h>
#elif defined(__GNUC__) || defined(__clang__)
// __builtin_cpu_init / __builtin_cpu_supports available via built-ins.
#endif

namespace world {
namespace simd {

// Function pointer storage. Default-initialized to scalar so any call before
// Initialize() still does the right thing.
SpectrumMultiplyHalfFn         SpectrumMultiplyHalf         = &scalar::SpectrumMultiplyHalf;
DcRemoveFn                     DcRemove                     = &scalar::DcRemove;
CompositeF0MergeFn             CompositeF0Merge             = &scalar::CompositeF0Merge;
ZcrSignChangeFn                ZcrSignChange                = &scalar::ZcrSignChange;
ZcrFrameSumFn                  ZcrFrameSum                  = &scalar::ZcrFrameSum;
ZeroCrossingDetectAndCollectFn ZeroCrossingDetectAndCollect = &scalar::ZeroCrossingDetectAndCollect;
ZeroCrossingIntervalsFn        ZeroCrossingIntervals        = &scalar::ZeroCrossingIntervals;
GaussianFilter1DInteriorFn     GaussianFilter1DInterior     = &scalar::GaussianFilter1DInterior;
FastNuttallWindow8Fn           FastNuttallWindow8           = &scalar::FastNuttallWindow8;
CosineModulateInPlaceFn        CosineModulateInPlace        = &scalar::CosineModulateInPlace;

namespace {

std::once_flag g_init_flag;
Tier g_active_tier = Tier::Scalar;
bool g_cpu_has_avx512 = false;

bool DetectAvx512() {
#if defined(__GNUC__) || defined(__clang__)
  __builtin_cpu_init();
  // Require F + DQ + BW + VL for the full mask/DQ/BW intrinsic surface.
  return __builtin_cpu_supports("avx512f")
      && __builtin_cpu_supports("avx512dq")
      && __builtin_cpu_supports("avx512bw")
      && __builtin_cpu_supports("avx512vl");
#elif defined(_MSC_VER)
  int info[4] = {0, 0, 0, 0};
  // Leaf 7, subleaf 0: extended features in EBX.
  __cpuidex(info, 7, 0);
  const unsigned ebx = static_cast<unsigned>(info[1]);
  const bool avx512f  = (ebx & (1u << 16)) != 0;
  const bool avx512dq = (ebx & (1u << 17)) != 0;
  const bool avx512bw = (ebx & (1u << 30)) != 0;
  const bool avx512vl = (ebx & (1u << 31)) != 0;
  // Check OS XSAVE support for ZMM state: XCR0 bits 5, 6, 7 (opmask, ZMM_Hi256, Hi16_ZMM).
  if (!(avx512f && avx512dq && avx512bw && avx512vl)) return false;
  // XGETBV requires OSXSAVE (CPUID.1:ECX bit 27).
  int cpuid1[4] = {0, 0, 0, 0};
  __cpuid(cpuid1, 1);
  const unsigned ecx1 = static_cast<unsigned>(cpuid1[2]);
  const bool osxsave = (ecx1 & (1u << 27)) != 0;
  if (!osxsave) return false;
  const unsigned long long xcr0 = _xgetbv(0);
  const unsigned long long zmm_mask = (1ull << 5) | (1ull << 6) | (1ull << 7);
  return (xcr0 & zmm_mask) == zmm_mask;
#else
  return false;
#endif
}

// Forward declarations for helpers defined below.
void ReadDisableMask(bool disable[10]);

void BindScalar() {
  SpectrumMultiplyHalf         = &scalar::SpectrumMultiplyHalf;
  DcRemove                     = &scalar::DcRemove;
  CompositeF0Merge             = &scalar::CompositeF0Merge;
  ZcrSignChange                = &scalar::ZcrSignChange;
  ZcrFrameSum                  = &scalar::ZcrFrameSum;
  ZeroCrossingDetectAndCollect = &scalar::ZeroCrossingDetectAndCollect;
  ZeroCrossingIntervals        = &scalar::ZeroCrossingIntervals;
  GaussianFilter1DInterior     = &scalar::GaussianFilter1DInterior;
  FastNuttallWindow8           = &scalar::FastNuttallWindow8;
  CosineModulateInPlace        = &scalar::CosineModulateInPlace;
  g_active_tier = Tier::Scalar;
}

#ifdef WORLD_HAS_AVX512
// Per-kernel disable state (reset each time BindAvx512 is called). Visible
// via GetDisabledMask() so bench can print which kernels are scalar.
bool g_disabled[10] = {false};

void BindAvx512() {
  ReadDisableMask(g_disabled);
  SpectrumMultiplyHalf         = g_disabled[0] ? &scalar::SpectrumMultiplyHalf         : &avx512::SpectrumMultiplyHalf;
  DcRemove                     = g_disabled[1] ? &scalar::DcRemove                     : &avx512::DcRemove;
  CompositeF0Merge             = g_disabled[2] ? &scalar::CompositeF0Merge             : &avx512::CompositeF0Merge;
  ZcrSignChange                = g_disabled[3] ? &scalar::ZcrSignChange                : &avx512::ZcrSignChange;
  ZcrFrameSum                  = g_disabled[4] ? &scalar::ZcrFrameSum                  : &avx512::ZcrFrameSum;
  ZeroCrossingDetectAndCollect = g_disabled[5] ? &scalar::ZeroCrossingDetectAndCollect : &avx512::ZeroCrossingDetectAndCollect;
  ZeroCrossingIntervals        = g_disabled[6] ? &scalar::ZeroCrossingIntervals        : &avx512::ZeroCrossingIntervals;
  GaussianFilter1DInterior     = g_disabled[7] ? &scalar::GaussianFilter1DInterior     : &avx512::GaussianFilter1DInterior;
  FastNuttallWindow8           = g_disabled[8] ? &scalar::FastNuttallWindow8           : &avx512::FastNuttallWindow8;
  CosineModulateInPlace        = g_disabled[9] ? &scalar::CosineModulateInPlace        : &avx512::CosineModulateInPlace;
  g_active_tier = Tier::AVX512;
}
#else
bool g_disabled[10] = {false};
#endif

const char *ReadForceEnv() {
#if defined(_MSC_VER)
  // getenv is deprecated on MSVC; use _dupenv_s.
  static char buf[32];
  buf[0] = '\0';
  size_t len = 0;
  char *tmp = nullptr;
  if (_dupenv_s(&tmp, &len, "WORLD_FORCE_SIMD_TIER") == 0 && tmp != nullptr) {
    size_t copy_len = len - 1;  // len includes trailing NUL
    if (copy_len >= sizeof(buf)) copy_len = sizeof(buf) - 1;
    memcpy(buf, tmp, copy_len);
    buf[copy_len] = '\0';
    free(tmp);
    return buf;
  }
  return nullptr;
#else
  return getenv("WORLD_FORCE_SIMD_TIER");
#endif
}

// Debug/bisect helper: env var WORLD_AVX512_DISABLE can list kernels (K1..K10)
// that should fall back to scalar even when AVX-512 tier is active. Syntax:
//   WORLD_AVX512_DISABLE=K1,K3,K6
// or: WORLD_AVX512_DISABLE=ALL   (force all to scalar even on AVX-512 tier)
// Use this to find which kernel is causing a correctness regression.
void ReadDisableMask(bool disable[10]) {
  for (int k = 0; k < 10; ++k) disable[k] = false;
  const char *env = nullptr;
#if defined(_MSC_VER)
  static char buf[128];
  buf[0] = '\0';
  size_t len = 0;
  char *tmp = nullptr;
  if (_dupenv_s(&tmp, &len, "WORLD_AVX512_DISABLE") == 0 && tmp != nullptr) {
    size_t copy_len = len - 1;
    if (copy_len >= sizeof(buf)) copy_len = sizeof(buf) - 1;
    memcpy(buf, tmp, copy_len);
    buf[copy_len] = '\0';
    free(tmp);
    env = buf;
  }
#else
  env = getenv("WORLD_AVX512_DISABLE");
#endif
  if (!env || !*env) return;
  // Check "ALL"
  if ((env[0] == 'A' || env[0] == 'a') && (env[1] == 'L' || env[1] == 'l') &&
      (env[2] == 'L' || env[2] == 'l')) {
    for (int k = 0; k < 10; ++k) disable[k] = true;
    return;
  }
  // Parse K<n>[,K<n>[,...]]
  const char *p = env;
  while (*p) {
    if (*p == 'K' || *p == 'k') {
      ++p;
      int kn = 0;
      while (*p >= '0' && *p <= '9') {
        kn = kn * 10 + (*p - '0');
        ++p;
      }
      if (kn >= 1 && kn <= 10) disable[kn - 1] = true;
    } else {
      ++p;
    }
  }
}

bool StringEqualsIgnoreCase(const char *a, const char *b) {
  if (!a || !b) return false;
  for (;;) {
    char ca = *a++, cb = *b++;
    if (ca >= 'A' && ca <= 'Z') ca = static_cast<char>(ca + 32);
    if (cb >= 'A' && cb <= 'Z') cb = static_cast<char>(cb + 32);
    if (ca != cb) return false;
    if (ca == '\0') return true;
  }
}

void DoInit() {
  g_cpu_has_avx512 = DetectAvx512();

  // Respect env override.
  const char *force = ReadForceEnv();
  if (force && *force) {
    if (StringEqualsIgnoreCase(force, "scalar")) {
      BindScalar();
      return;
    }
    if (StringEqualsIgnoreCase(force, "avx512")) {
#ifdef WORLD_HAS_AVX512
      if (g_cpu_has_avx512) {
        BindAvx512();
        return;
      }
#endif
      BindScalar();
      return;
    }
    // Unknown value: fall through to auto-detect.
  }

#ifdef WORLD_HAS_AVX512
  if (g_cpu_has_avx512) {
    BindAvx512();
    return;
  }
#endif
  BindScalar();
}

}  // namespace

void Initialize() {
  std::call_once(g_init_flag, DoInit);
}

Tier GetActiveTier() {
  Initialize();
  return g_active_tier;
}

const char *GetActiveTierName() {
  Initialize();
  switch (g_active_tier) {
    case Tier::Scalar: return "Scalar";
    case Tier::AVX512: return "AVX512";
  }
  return "Unknown";
}

bool CpuHasAvx512() {
  Initialize();
  return g_cpu_has_avx512;
}

const char *GetDisabledKernels() {
  Initialize();
  static char buf[64];
  buf[0] = '\0';
  int pos = 0;
  for (int k = 0; k < 10; ++k) {
    if (!g_disabled[k]) continue;
    if (pos > 0 && pos < static_cast<int>(sizeof(buf)) - 1) buf[pos++] = ',';
    const int kn = k + 1;
    if (kn < 10) {
      if (pos + 2 < static_cast<int>(sizeof(buf))) {
        buf[pos++] = 'K';
        buf[pos++] = static_cast<char>('0' + kn);
      }
    } else {
      if (pos + 3 < static_cast<int>(sizeof(buf))) {
        buf[pos++] = 'K';
        buf[pos++] = '1';
        buf[pos++] = '0';
      }
    }
  }
  buf[pos] = '\0';
  return buf;
}

void ForceTier(Tier tier) {
  Initialize();  // detect CPU first
  switch (tier) {
    case Tier::AVX512:
#ifdef WORLD_HAS_AVX512
      if (g_cpu_has_avx512) {
        BindAvx512();
        return;
      }
#endif
      // Fall through to scalar if unavailable.
      BindScalar();
      return;
    case Tier::Scalar:
    default:
      BindScalar();
      return;
  }
}

}  // namespace simd
}  // namespace world
