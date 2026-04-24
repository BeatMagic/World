//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// SIMD kernel API: function-pointer types + scalar/AVX-512 implementation
// prototypes for the performance-critical inner loops used by CompositeF0,
// Harvest, DIO, and ZCR.
//
// The dispatcher (simd_dispatch.h) binds one extern function pointer per
// kernel, so call sites call `world::simd::SpectrumMultiplyHalf(...)` without
// knowing which tier is active.
//-----------------------------------------------------------------------------
#ifndef WORLD_SIMD_KERNELS_H_
#define WORLD_SIMD_KERNELS_H_

#include "world/fft.h"

namespace world {
namespace simd {

// Function-pointer types --------------------------------------------------

// K1. Complex spectrum multiplication: out[i] = a[i] * b[i] for i in [0, half_plus_1).
//     fft_complex is double[2]; out may alias either input.
using SpectrumMultiplyHalfFn = void (*)(const fft_complex *a,
    const fft_complex *b, fft_complex *out, int half_plus_1);

// K2. Remove mean in place: mean = avg(y[0..n)); y[i] -= mean.
using DcRemoveFn = void (*)(double *y, int n);

// K3. Composite F0 voicing merge (compositef0.cpp:185-191):
//     for i in [0, f0_length):
//       if i < effective_zcr_length && zcr_d_a[i] >= threshold:
//         f0_out[i] = (f0_dio[i] > 0.0) ? f0_harvest[i] : 0.0;
//       else:
//         f0_out[i] = f0_harvest[i];
using CompositeF0MergeFn = void (*)(const double *f0_dio,
    const double *f0_harvest, const double *zcr_d_a, int f0_length,
    int effective_zcr_length, double threshold, double *f0_out);

// K4. ZCR sign-change detection (compositef0.cpp:62-66):
//     for i in [0, n_minus_1):
//       crossings_out[i] = ((x[i] < 0) != (x[i+1] < 0)) ? 1.0 : 0.0;
using ZcrSignChangeFn = void (*)(const double *x, int n_minus_1,
    double *crossings_out);

// K5. ZCR frame-wise sum (compositef0.cpp:71-79):
//     Matches librosa zero_crossing_rate with center=True, pad=True.
//     Position 0 of every frame always counts as 1.0.
using ZcrFrameSumFn = void (*)(const double *crossings, int total_len,
    int hop_length, int frame_length, int zcr_length, double *zcr_out);

// K6. Zero-crossing detect + index compaction (harvest.cpp:192-200, dio.cpp:372-380).
//     Finds all i in [0, y_length-1) where filtered_signal[i] > 0 AND
//     filtered_signal[i+1] <= 0. Writes (i+1) to edges_out in ascending order,
//     sets *count_out to the number of edges found.
using ZeroCrossingDetectAndCollectFn = void (*)(const double *filtered_signal,
    int y_length, int *edges_out, int *count_out);

// K7. Zero-crossing intervals + locations (harvest.cpp:208-211, dio.cpp:389-392):
//     for i in [0, count-1):
//       intervals[i]  = fs / (fine_edges[i+1] - fine_edges[i]);
//       locations[i]  = (fine_edges[i] + fine_edges[i+1]) * 0.5 / fs;
using ZeroCrossingIntervalsFn = void (*)(const double *fine_edges, int count,
    double fs, double *intervals, double *locations);

// K8. Gaussian 1D convolution interior only (compositef0.cpp:108-121).
//     Computes y[i] = sum_{j=0..2*radius} x[i+j-radius] * kernel[j] for
//     i in [radius, x_length-radius). Caller handles reflect boundaries.
using GaussianFilter1DInteriorFn = void (*)(const double *x, int x_length,
    const double *kernel, int radius, double *y);

// K9. Fast Nuttall window (harvest.cpp:100-119, dio.cpp:41-60):
//     y[i] = 0.355768 - 0.487396*cos(w*i) + 0.144232*cos(2*w*i) -
//            0.012604*cos(3*w*i), where w = 2*pi/(y_length-1).
//     Special case y_length <= 1 handled internally.
using FastNuttallWindow8Fn = void (*)(int y_length, double *y);

// K10. Cosine modulation in place (harvest.cpp:134-143):
//      waveform[i] *= cos(start_phase + i * w_step) for i in [0, length).
using CosineModulateInPlaceFn = void (*)(double *waveform, int length,
    double start_phase, double w_step);

// Scalar implementations (always available; form the Scalar tier) ----------
namespace scalar {

void SpectrumMultiplyHalf(const fft_complex *a, const fft_complex *b,
    fft_complex *out, int half_plus_1);
void DcRemove(double *y, int n);
void CompositeF0Merge(const double *f0_dio, const double *f0_harvest,
    const double *zcr_d_a, int f0_length, int effective_zcr_length,
    double threshold, double *f0_out);
void ZcrSignChange(const double *x, int n_minus_1, double *crossings_out);
void ZcrFrameSum(const double *crossings, int total_len, int hop_length,
    int frame_length, int zcr_length, double *zcr_out);
void ZeroCrossingDetectAndCollect(const double *filtered_signal, int y_length,
    int *edges_out, int *count_out);
void ZeroCrossingIntervals(const double *fine_edges, int count, double fs,
    double *intervals, double *locations);
void GaussianFilter1DInterior(const double *x, int x_length,
    const double *kernel, int radius, double *y);
void FastNuttallWindow8(int y_length, double *y);
void CosineModulateInPlace(double *waveform, int length, double start_phase,
    double w_step);

}  // namespace scalar

// AVX-512 implementations (only linked when WORLD_HAS_AVX512 is defined) ---
// Phase 0: each of these delegates to scalar. Phase 1-3 replaces with
// real AVX-512 intrinsics.
#ifdef WORLD_HAS_AVX512
namespace avx512 {

void SpectrumMultiplyHalf(const fft_complex *a, const fft_complex *b,
    fft_complex *out, int half_plus_1);
void DcRemove(double *y, int n);
void CompositeF0Merge(const double *f0_dio, const double *f0_harvest,
    const double *zcr_d_a, int f0_length, int effective_zcr_length,
    double threshold, double *f0_out);
void ZcrSignChange(const double *x, int n_minus_1, double *crossings_out);
void ZcrFrameSum(const double *crossings, int total_len, int hop_length,
    int frame_length, int zcr_length, double *zcr_out);
void ZeroCrossingDetectAndCollect(const double *filtered_signal, int y_length,
    int *edges_out, int *count_out);
void ZeroCrossingIntervals(const double *fine_edges, int count, double fs,
    double *intervals, double *locations);
void GaussianFilter1DInterior(const double *x, int x_length,
    const double *kernel, int radius, double *y);
void FastNuttallWindow8(int y_length, double *y);
void CosineModulateInPlace(double *waveform, int length, double start_phase,
    double w_step);

}  // namespace avx512
#endif  // WORLD_HAS_AVX512

}  // namespace simd
}  // namespace world

#endif  // WORLD_SIMD_KERNELS_H_
