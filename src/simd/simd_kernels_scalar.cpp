//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// Scalar implementations of the CompositeF0-path SIMD kernels. These are the
// Tier-Scalar fallback AND the reference against which AVX-512 kernels are
// validated. The logic is copied verbatim from harvest.cpp / dio.cpp /
// compositef0.cpp so that baseline bit-identity is preserved.
//-----------------------------------------------------------------------------
#include "simd/simd_kernels.h"

#include <math.h>

#include "world/constantnumbers.h"

namespace world {
namespace simd {
namespace scalar {

// K1. Complex spectrum multiplication --------------------------------------
void SpectrumMultiplyHalf(const fft_complex *a, const fft_complex *b,
    fft_complex *out, int half_plus_1) {
  for (int i = 0; i < half_plus_1; ++i) {
    const double tmp = a[i][0] * b[i][0] - a[i][1] * b[i][1];
    out[i][1] = a[i][0] * b[i][1] + a[i][1] * b[i][0];
    out[i][0] = tmp;
  }
}

// K2. DC removal in place --------------------------------------------------
void DcRemove(double *y, int n) {
  double mean_y = 0.0;
  for (int i = 0; i < n; ++i) mean_y += y[i];
  mean_y /= n;
  for (int i = 0; i < n; ++i) y[i] -= mean_y;
}

// K3. Composite F0 voicing merge -------------------------------------------
void CompositeF0Merge(const double *f0_dio, const double *f0_harvest,
    const double *zcr_d_a, int f0_length, int effective_zcr_length,
    double threshold, double *f0_out) {
  for (int i = 0; i < f0_length; ++i) {
    if (i < effective_zcr_length && zcr_d_a[i] >= threshold) {
      f0_out[i] = (f0_dio[i] > 0.0) ? f0_harvest[i] : 0.0;
    } else {
      f0_out[i] = f0_harvest[i];
    }
  }
}

// K4. ZCR sign-change detection --------------------------------------------
void ZcrSignChange(const double *x, int n_minus_1, double *crossings_out) {
  for (int i = 0; i < n_minus_1; ++i) {
    const bool sign_curr = x[i] < 0.0;
    const bool sign_next = x[i + 1] < 0.0;
    crossings_out[i] = (sign_curr != sign_next) ? 1.0 : 0.0;
  }
}

// K5. ZCR frame-wise sum ---------------------------------------------------
// Bit-identical to compositef0.cpp:71-79 -- uses explicit division (not
// multiply-by-reciprocal) so baseline regression tests stay exact.
void ZcrFrameSum(const double *crossings, int total_len, int hop_length,
    int frame_length, int zcr_length, double *zcr_out) {
  for (int frame = 0; frame < zcr_length; ++frame) {
    const int start = frame * hop_length;
    double sum = 1.0;  // pad=True: position 0 always counts
    for (int i = 1; i < frame_length; ++i) {
      const int idx = start + i;
      if (idx < total_len) sum += crossings[idx];
    }
    zcr_out[frame] = sum / frame_length;
  }
}

// K6. Zero-crossing detect + index compaction ------------------------------
void ZeroCrossingDetectAndCollect(const double *filtered_signal, int y_length,
    int *edges_out, int *count_out) {
  int count = 0;
  // Match original two-pass behavior exactly: first fills negative_going_points
  // then compacts. Combined single pass is semantically identical since the
  // compact preserves ascending order.
  for (int i = 0; i < y_length - 1; ++i) {
    if (0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0) {
      edges_out[count++] = i + 1;
    }
  }
  *count_out = count;
}

// K7. Zero-crossing intervals ---------------------------------------------
void ZeroCrossingIntervals(const double *fine_edges, int count, double fs,
    double *intervals, double *locations) {
  for (int i = 0; i < count - 1; ++i) {
    intervals[i] = fs / (fine_edges[i + 1] - fine_edges[i]);
    locations[i] = (fine_edges[i] + fine_edges[i + 1]) / 2.0 / fs;
  }
}

// K8. Gaussian 1D interior convolution -------------------------------------
void GaussianFilter1DInterior(const double *x, int x_length,
    const double *kernel, int radius, double *y) {
  const int kernel_length = 2 * radius + 1;
  const int interior_begin = radius;
  const int interior_end = x_length - radius;
  for (int i = interior_begin; i < interior_end; ++i) {
    double val = 0.0;
    for (int j = 0; j < kernel_length; ++j) {
      val += x[i + j - radius] * kernel[j];
    }
    y[i] = val;
  }
}

// K9. Fast Nuttall window (via 3 recursive oscillators) --------------------
// Bit-identical to harvest.cpp:100-119 / dio.cpp:41-60.
void FastNuttallWindow8(int y_length, double *y) {
  if (y_length <= 1) {
    if (y_length == 1) y[0] = 1.0;
    return;
  }
  const double phase_step = 2.0 * world::kPi / (y_length - 1);
  const double cos1 = cos(phase_step), sin1 = sin(phase_step);
  const double cos2 = cos(2.0 * phase_step), sin2 = sin(2.0 * phase_step);
  const double cos3 = cos(3.0 * phase_step), sin3 = sin(3.0 * phase_step);
  double c1 = 1.0, s1 = 0.0;
  double c2 = 1.0, s2 = 0.0;
  double c3 = 1.0, s3 = 0.0;
  for (int i = 0; i < y_length; ++i) {
    y[i] = 0.355768 - 0.487396 * c1 + 0.144232 * c2 - 0.012604 * c3;
    double nc;
    nc = c1 * cos1 - s1 * sin1; s1 = s1 * cos1 + c1 * sin1; c1 = nc;
    nc = c2 * cos2 - s2 * sin2; s2 = s2 * cos2 + c2 * sin2; c2 = nc;
    nc = c3 * cos3 - s3 * sin3; s3 = s3 * cos3 + c3 * sin3; c3 = nc;
  }
}

// K10. Cosine modulation in place (via recursive oscillator) ---------------
// Bit-identical to harvest.cpp:134-143. waveform[i] *= cos(start_phase + i*w_step).
void CosineModulateInPlace(double *waveform, int length, double start_phase,
    double w_step) {
  if (length <= 0) return;
  const double cos_w = cos(w_step);
  const double sin_w = sin(w_step);
  double c = cos(start_phase);
  double s = sin(start_phase);
  for (int i = 0; i < length; ++i) {
    waveform[i] *= c;
    const double nc = c * cos_w - s * sin_w;
    s = s * cos_w + c * sin_w;
    c = nc;
  }
}

}  // namespace scalar
}  // namespace simd
}  // namespace world
