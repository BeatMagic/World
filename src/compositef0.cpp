//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// CompositeF0 combines DIO and Harvest F0 estimators with zero-crossing rate
// analysis for more robust voiced/unvoiced decision.
//-----------------------------------------------------------------------------
#include "world/compositef0.h"

#include <math.h>
#include <string.h>

#include "simd/simd_dispatch.h"
#include "world/common.h"
#include "world/constantnumbers.h"
#include "world/dio.h"
#include "world/harvest.h"
#include "world/matlabfunctions.h"

namespace {

//-----------------------------------------------------------------------------
// DeSuddenChange() fills in 1-2 frame gaps of zero in the F0 contour.
// A single zero frame between two voiced frames is interpolated.
// Two consecutive zero frames between voiced frames are also interpolated.
//-----------------------------------------------------------------------------
static void DeSuddenChange(double *f0, int f0_length) {
  for (int i = 1; i < f0_length - 2; ++i) {
    if (f0[i - 1] != 0.0 && f0[i] == 0.0) {
      if (f0[i + 1] != 0.0) {
        f0[i] = (f0[i - 1] + f0[i + 1]) / 2.0;
      } else if (f0[i + 2] != 0.0) {
        f0[i] = f0[i + 1] = (f0[i - 1] + f0[i + 2]) / 2.0;
      }
    }
  }
}

//-----------------------------------------------------------------------------
// ComputeZeroCrossingRate() computes the zero-crossing rate per frame,
// matching librosa.feature.zero_crossing_rate with center=True, pad=True.
//
// When called with pad=True (the default in Python reference), librosa's
// zero_crossings sets position 0 of every frame to True (always a crossing).
// In librosa 0.11+, zero_crossing_rate has no explicit 'pad' parameter, so
// pad=True flows through **kwargs into zero_crossings unchanged.
//
// Approach:
//   1. Compute per-sample crossings on raw signal (unsigned: 1 if sign change)
//   2. Center-pad the crossings array with zeros (frame_length/2 each side)
//   3. Frame and compute mean: position 0 always counts as 1 (pad=True),
//      positions 1..frame_length-1 use real crossings
//-----------------------------------------------------------------------------
static void ComputeZeroCrossingRate(const double *x, int x_length,
    int frame_length, int hop_length, double *zcr, int zcr_length) {
  int raw_len = x_length - 1;

  int pad_length = frame_length / 2;
  // Layout: [zeros * pad_length] [prepend 0] [raw crossings] [zeros * pad_length]
  int total_len = pad_length + 1 + raw_len + pad_length;
  double *crossings = new double[total_len];
  memset(crossings, 0, total_len * sizeof(double));

  // K4: sign-change detect. Writes into crossings[pad_length+1..pad_length+raw_len].
  world::simd::ZcrSignChange(x, raw_len, crossings + pad_length + 1);

  // K5: frame crossings and compute mean.
  // Position 0 of every frame is always 1.0 (matching librosa pad=True).
  // Positions 1..frame_length-1 use the real crossings from the global array.
  world::simd::ZcrFrameSum(crossings, total_len, hop_length, frame_length,
                           zcr_length, zcr);

  delete[] crossings;
}

//-----------------------------------------------------------------------------
// GaussianFilter1D() applies 1D Gaussian smoothing, matching
// scipy.ndimage.gaussian_filter1d with default truncate=4.0 and
// mode='reflect'.
//
// Kernel radius = int(4.0 * sigma + 0.5), direct convolution.
//-----------------------------------------------------------------------------
static void GaussianFilter1D(const double *x, int x_length, double sigma,
    double *y) {
  int radius = static_cast<int>(4.0 * sigma + 0.5);
  int kernel_length = 2 * radius + 1;
  double *kernel = new double[kernel_length];

  // Compute normalized Gaussian kernel
  double sum = 0.0;
  for (int i = 0; i < kernel_length; ++i) {
    double t = static_cast<double>(i - radius);
    kernel[i] = exp(-0.5 * t * t / (sigma * sigma));
    sum += kernel[i];
  }
  for (int i = 0; i < kernel_length; ++i) kernel[i] /= sum;

  // Convolution with reflect boundary (scipy 'reflect' mode:
  // index -1 maps to 1, -2 maps to 2, etc.)
  //
  // Split into three regions:
  //   Left boundary  [0, radius)                    -- scalar with reflect
  //   Interior       [radius, x_length - radius)    -- K8 dispatch (no reflect needed)
  //   Right boundary [x_length - radius, x_length)  -- scalar with reflect
  // Fallback to full scalar when x_length is too small for interior to exist.
  if (x_length <= 2 * radius) {
    for (int i = 0; i < x_length; ++i) {
      double val = 0.0;
      for (int j = 0; j < kernel_length; ++j) {
        int idx = i + j - radius;
        if (idx < 0) idx = -idx;
        if (idx >= x_length) idx = 2 * (x_length - 1) - idx;
        if (idx < 0) idx = 0;
        if (idx >= x_length) idx = x_length - 1;
        val += x[idx] * kernel[j];
      }
      y[i] = val;
    }
  } else {
    // Left boundary
    for (int i = 0; i < radius; ++i) {
      double val = 0.0;
      for (int j = 0; j < kernel_length; ++j) {
        int idx = i + j - radius;
        if (idx < 0) idx = -idx;
        if (idx >= x_length) idx = 2 * (x_length - 1) - idx;
        if (idx < 0) idx = 0;
        if (idx >= x_length) idx = x_length - 1;
        val += x[idx] * kernel[j];
      }
      y[i] = val;
    }
    // Interior
    world::simd::GaussianFilter1DInterior(x, x_length, kernel, radius, y);
    // Right boundary
    for (int i = x_length - radius; i < x_length; ++i) {
      double val = 0.0;
      for (int j = 0; j < kernel_length; ++j) {
        int idx = i + j - radius;
        if (idx < 0) idx = -idx;
        if (idx >= x_length) idx = 2 * (x_length - 1) - idx;
        if (idx < 0) idx = 0;
        if (idx >= x_length) idx = x_length - 1;
        val += x[idx] * kernel[j];
      }
      y[i] = val;
    }
  }

  delete[] kernel;
}

}  // namespace

int GetSamplesForCompositeF0(int fs, int x_length, double frame_period) {
  return static_cast<int>(1000.0 * x_length / fs / frame_period) + 1;
}

void CompositeF0(const double *x, int x_length, int fs,
    const CompositeF0Option *option, double *temporal_positions, double *f0) {
  // Detect CPU + bind SIMD kernel function pointers. Thread-safe; no-op on
  // subsequent calls.
  world::simd::Initialize();

  double frame_period = option->frame_period;
  int f0_length = GetSamplesForCompositeF0(fs, x_length, frame_period);
  int hop_length = matlab_round(frame_period / 1000.0 * fs);

  // Step 1: Run DIO
  DioOption dio_option = {0};
  InitializeDioOption(&dio_option);
  dio_option.f0_floor = option->f0_floor;
  dio_option.f0_ceil = option->f0_ceil;
  dio_option.frame_period = frame_period;
  double *f0_dio = new double[f0_length];
  double *temporal_dio = new double[f0_length];
  Dio(x, x_length, fs, &dio_option, temporal_dio, f0_dio);

  // Step 2: Apply DeSuddenChange to DIO result
  DeSuddenChange(f0_dio, f0_length);

  // Step 3: Run Harvest
  HarvestOption harvest_option = {0};
  InitializeHarvestOption(&harvest_option);
  harvest_option.f0_floor = option->f0_floor;
  harvest_option.f0_ceil = option->f0_ceil;
  harvest_option.frame_period = frame_period;
  double *f0_harvest = new double[f0_length];
  Harvest(x, x_length, fs, &harvest_option, temporal_positions, f0_harvest);

  // Step 4: Compute ZCR
  int zcr_length = 1 + (x_length - 1) / hop_length;
  double *zcr = new double[zcr_length];
  ComputeZeroCrossingRate(x, x_length, option->zcr_frame_length,
      hop_length, zcr, zcr_length);

  // Step 5: Compute |diff(zcr)| with reflect pad on last element
  double *zcr_diff = new double[zcr_length];
  for (int i = 0; i < zcr_length - 1; ++i)
    zcr_diff[i] = fabs(zcr[i + 1] - zcr[i]);
  // np.pad(..., [0, 1], mode="reflect") → last element = second-to-last
  zcr_diff[zcr_length - 1] = (zcr_length >= 2) ?
      zcr_diff[zcr_length - 2] : 0.0;

  // Step 6: Apply Gaussian filter
  double *zcr_d_a = new double[zcr_length];
  GaussianFilter1D(zcr_diff, zcr_length, option->gaussian_sigma, zcr_d_a);

  // Step 7: Handle length alignment
  // Python: if len(zcr_d_a) == len(f0_harvest) + 1: zcr_d_a = zcr_d_a[:-1]
  int effective_zcr_length = zcr_length;
  if (zcr_length == f0_length + 1) effective_zcr_length = f0_length;

  // Step 8: Composite voicing decision (K3)
  // np.where(zcr_d_a < 0.002, f0_harvest, np.where(f0_dio > 0, f0_harvest, 0))
  world::simd::CompositeF0Merge(f0_dio, f0_harvest, zcr_d_a, f0_length,
                                effective_zcr_length,
                                option->zcr_threshold, f0);

  delete[] f0_dio;
  delete[] temporal_dio;
  delete[] f0_harvest;
  delete[] zcr;
  delete[] zcr_diff;
  delete[] zcr_d_a;
}

void InitializeCompositeF0Option(CompositeF0Option *option) {
  option->f0_floor = 50.0;
  option->f0_ceil = 1100.0;
  option->frame_period = 5.0;
  option->zcr_frame_length = 2048;
  option->zcr_threshold = 0.002;
  option->gaussian_sigma = 4.0;
}
