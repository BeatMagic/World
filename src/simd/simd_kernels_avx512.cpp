//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// AVX-512 implementations of CompositeF0-path SIMD kernels.
//
// Compiled with /arch:AVX512 (MSVC) or -mavx512f+DQ+BW+VL+FMA+BMI2 (GCC/Clang).
// Other translation units do NOT get those flags; all AVX-512 intrinsic code
// must live in this file.
//
// When WORLD_HAS_AVX512 is not defined (compiler lacks AVX-512 support), the
// file becomes a no-op (the dispatcher binds scalar in that case).
//
// All kernels are tuned for AMD Zen5 (znver5):
//   - Two full 512-bit FMA ports (no double-pumping like Zen4)
//   - 4-wide AVX-512 execution per cycle on basic ops
//   - Masking via k-registers eliminates branches
//   - Hardware gather/scatter have improved latency
//
// Correctness requirement: output of each avx512:: kernel must match the
// scalar reference (scalar::) within <= 4 ULP for double. Where bit-exact
// matters (e.g. FMA vs separate mul+add reordering in the final-accumulate
// phase), the intrinsic is chosen to preserve the scalar's operation order.
//-----------------------------------------------------------------------------
#include "simd/simd_kernels.h"

#ifdef WORLD_HAS_AVX512

#include <immintrin.h>
#include <math.h>

#include "world/constantnumbers.h"

namespace world {
namespace simd {
namespace avx512 {

// K1. Complex spectrum multiplication -------------------------------------
// Version 3: explicit add/sub + mask_blend (no fmsubadd).
//
// Processes 4 complex pairs (8 doubles) per iteration with interleaved layout
// preserved. Uses permutexvar_pd with explicit index vectors and an explicit
// diff/sum computation with mask_blend, avoiding fmsubadd entirely.
//
// Per iteration:
//   prod_lo = {ar*br, ar*bi, ar*br, ar*bi, ...}   (va_re element-wise vb)
//   prod_hi = {ai*bi, ai*br, ai*bi, ai*br, ...}   (va_im element-wise vb_swap)
//   diff    = prod_lo - prod_hi    (correct at even lanes: ar*br - ai*bi)
//   sum     = prod_lo + prod_hi    (correct at odd lanes:  ar*bi + ai*br)
//   result  = blend(odd_mask=0xAA, diff, sum)
void SpectrumMultiplyHalf(const fft_complex *a, const fft_complex *b,
    fft_complex *out, int half_plus_1) {
  const double *pa = &a[0][0];
  const double *pb = &b[0][0];
  double *po = &out[0][0];

  const __m512i idx_re_bcast = _mm512_setr_epi64(0, 0, 2, 2, 4, 4, 6, 6);
  const __m512i idx_im_bcast = _mm512_setr_epi64(1, 1, 3, 3, 5, 5, 7, 7);
  const __m512i idx_swap     = _mm512_setr_epi64(1, 0, 3, 2, 5, 4, 7, 6);

  int i = 0;
  const int simd_end = half_plus_1 & ~3;
  for (; i < simd_end; i += 4) {
    const __m512d va = _mm512_loadu_pd(pa + 2 * i);
    const __m512d vb = _mm512_loadu_pd(pb + 2 * i);
    const __m512d va_re   = _mm512_permutexvar_pd(idx_re_bcast, va);
    const __m512d va_im   = _mm512_permutexvar_pd(idx_im_bcast, va);
    const __m512d vb_swap = _mm512_permutexvar_pd(idx_swap, vb);
    const __m512d prod_lo = _mm512_mul_pd(va_re, vb);
    const __m512d prod_hi = _mm512_mul_pd(va_im, vb_swap);
    const __m512d diff    = _mm512_sub_pd(prod_lo, prod_hi);
    const __m512d sum     = _mm512_add_pd(prod_lo, prod_hi);
    // Odd lanes (1,3,5,7) -> take sum; even lanes (0,2,4,6) -> take diff.
    const __mmask8 odd_mask = 0xAA;  // 0b10101010
    const __m512d result = _mm512_mask_blend_pd(odd_mask, diff, sum);
    _mm512_storeu_pd(po + 2 * i, result);
  }
  for (; i < half_plus_1; ++i) {
    const double tmp = pa[2*i] * pb[2*i] - pa[2*i+1] * pb[2*i+1];
    po[2*i+1] = pa[2*i] * pb[2*i+1] + pa[2*i+1] * pb[2*i];
    po[2*i] = tmp;
  }
}

// K2. DC removal in place --------------------------------------------------
// Two-pass: (1) horizontal sum -> mean; (2) broadcast subtract.
// Unrolled 4x for ILP on the reduction pass (hides FMA/add latency ~4c).
void DcRemove(double *y, int n) {
  __m512d acc0 = _mm512_setzero_pd();
  __m512d acc1 = _mm512_setzero_pd();
  __m512d acc2 = _mm512_setzero_pd();
  __m512d acc3 = _mm512_setzero_pd();

  int i = 0;
  const int simd_end = n & ~31;  // 32 doubles per unrolled iter
  for (; i < simd_end; i += 32) {
    acc0 = _mm512_add_pd(acc0, _mm512_loadu_pd(y + i +  0));
    acc1 = _mm512_add_pd(acc1, _mm512_loadu_pd(y + i +  8));
    acc2 = _mm512_add_pd(acc2, _mm512_loadu_pd(y + i + 16));
    acc3 = _mm512_add_pd(acc3, _mm512_loadu_pd(y + i + 24));
  }
  // Fold and horizontal reduce.
  const __m512d acc = _mm512_add_pd(_mm512_add_pd(acc0, acc1),
                                    _mm512_add_pd(acc2, acc3));
  double mean_y = _mm512_reduce_add_pd(acc);
  // Scalar tail of partial sum.
  for (; i < n; ++i) mean_y += y[i];
  mean_y /= n;

  // Broadcast subtract.
  const __m512d vmean = _mm512_set1_pd(mean_y);
  i = 0;
  for (; i < simd_end; i += 32) {
    _mm512_storeu_pd(y + i +  0, _mm512_sub_pd(_mm512_loadu_pd(y + i +  0), vmean));
    _mm512_storeu_pd(y + i +  8, _mm512_sub_pd(_mm512_loadu_pd(y + i +  8), vmean));
    _mm512_storeu_pd(y + i + 16, _mm512_sub_pd(_mm512_loadu_pd(y + i + 16), vmean));
    _mm512_storeu_pd(y + i + 24, _mm512_sub_pd(_mm512_loadu_pd(y + i + 24), vmean));
  }
  for (; i < n; ++i) y[i] -= mean_y;
}

// K3. Composite F0 voicing merge -------------------------------------------
// Branchless implementation of compositef0.cpp:185-191:
//   for i in [0, f0_length):
//     if i < effective_zcr_length && zcr_d_a[i] >= threshold:
//       f0[i] = (f0_dio[i] > 0) ? f0_harvest[i] : 0
//     else:
//       f0[i] = f0_harvest[i]
// Equivalent: f0[i] = (valid & zcr_high & !dio_voiced) ? 0 : f0_harvest[i]
void CompositeF0Merge(const double *f0_dio, const double *f0_harvest,
    const double *zcr_d_a, int f0_length, int effective_zcr_length,
    double threshold, double *f0_out) {
  const __m512d vthr = _mm512_set1_pd(threshold);
  const __m512d vzero = _mm512_setzero_pd();

  int i = 0;
  // Fast path: fully inside effective_zcr_length (no per-lane valid-range mask).
  const int fast_end = (effective_zcr_length < f0_length
                        ? effective_zcr_length : f0_length) & ~7;
  for (; i < fast_end; i += 8) {
    const __m512d vdio = _mm512_loadu_pd(f0_dio + i);
    const __m512d vhar = _mm512_loadu_pd(f0_harvest + i);
    const __m512d vzcr = _mm512_loadu_pd(zcr_d_a + i);
    const __mmask8 zcr_high    = _mm512_cmp_pd_mask(vzcr, vthr, _CMP_GE_OQ);
    const __mmask8 dio_voiced  = _mm512_cmp_pd_mask(vdio, vzero, _CMP_GT_OQ);
    // Lanes to zero out: zcr_high AND NOT dio_voiced.
    const __mmask8 zero_mask   = zcr_high & ~dio_voiced;
    // Otherwise keep f0_harvest.
    const __m512d result = _mm512_mask_blend_pd(zero_mask, vhar, vzero);
    _mm512_storeu_pd(f0_out + i, result);
  }
  // Tail: either within effective range with remainder, or past effective range.
  for (; i < f0_length; ++i) {
    if (i < effective_zcr_length && zcr_d_a[i] >= threshold) {
      f0_out[i] = (f0_dio[i] > 0.0) ? f0_harvest[i] : 0.0;
    } else {
      f0_out[i] = f0_harvest[i];
    }
  }
}

// K4. ZCR sign-change detection --------------------------------------------
// crossings_out[i] = ((x[i] < 0) != (x[i+1] < 0)) ? 1.0 : 0.0
// Eliminates the unpredictable branch in the scalar version.
void ZcrSignChange(const double *x, int n_minus_1, double *crossings_out) {
  const __m512d vzero = _mm512_setzero_pd();
  const __m512d vone  = _mm512_set1_pd(1.0);

  int i = 0;
  const int simd_end = n_minus_1 & ~7;
  for (; i < simd_end; i += 8) {
    const __m512d x_curr = _mm512_loadu_pd(x + i);
    const __m512d x_next = _mm512_loadu_pd(x + i + 1);
    const __mmask8 m_curr = _mm512_cmp_pd_mask(x_curr, vzero, _CMP_LT_OQ);
    const __mmask8 m_next = _mm512_cmp_pd_mask(x_next, vzero, _CMP_LT_OQ);
    const __mmask8 diff   = m_curr ^ m_next;
    const __m512d out = _mm512_mask_blend_pd(diff, vzero, vone);
    _mm512_storeu_pd(crossings_out + i, out);
  }
  // Scalar tail.
  for (; i < n_minus_1; ++i) {
    const bool sc = x[i] < 0.0;
    const bool sn = x[i + 1] < 0.0;
    crossings_out[i] = (sc != sn) ? 1.0 : 0.0;
  }
}

// K5. ZCR frame-wise sum ---------------------------------------------------
// Per-frame horizontal reduction. Fast path handles frames fully within
// `total_len` with no per-element boundary check. Slow path (tail frames)
// keeps the scalar reference.
void ZcrFrameSum(const double *crossings, int total_len, int hop_length,
    int frame_length, int zcr_length, double *zcr_out) {
  const double fl_d = static_cast<double>(frame_length);
  for (int frame = 0; frame < zcr_length; ++frame) {
    const int start = frame * hop_length;
    const int last_idx = start + frame_length - 1;
    double sum = 1.0;  // pad=True: position 0 always counts
    if (last_idx < total_len) {
      // Fast path: entire frame within bounds; pure SIMD reduction from i=1.
      __m512d a0 = _mm512_setzero_pd();
      __m512d a1 = _mm512_setzero_pd();
      __m512d a2 = _mm512_setzero_pd();
      __m512d a3 = _mm512_setzero_pd();
      int i = 1;
      const int base = start + 1;
      // Remaining count after skipping i=0: frame_length - 1.
      const int count_simd = (frame_length - 1) & ~31;  // multiples of 32
      const int simd_end = i + count_simd;
      for (; i < simd_end; i += 32) {
        a0 = _mm512_add_pd(a0, _mm512_loadu_pd(crossings + base + i - 1 +  0));
        a1 = _mm512_add_pd(a1, _mm512_loadu_pd(crossings + base + i - 1 +  8));
        a2 = _mm512_add_pd(a2, _mm512_loadu_pd(crossings + base + i - 1 + 16));
        a3 = _mm512_add_pd(a3, _mm512_loadu_pd(crossings + base + i - 1 + 24));
      }
      const __m512d acc = _mm512_add_pd(_mm512_add_pd(a0, a1),
                                        _mm512_add_pd(a2, a3));
      sum += _mm512_reduce_add_pd(acc);
      // Scalar tail.
      for (; i < frame_length; ++i) sum += crossings[start + i];
    } else {
      // Slow path: boundary check per element.
      for (int i = 1; i < frame_length; ++i) {
        const int idx = start + i;
        if (idx < total_len) sum += crossings[idx];
      }
    }
    zcr_out[frame] = sum / fl_d;
  }
}

// K6. Zero-crossing detect + index compaction -----------------------------
// For each i in [0, y_length-1), emit (i+1) to edges_out if:
//   filtered_signal[i] > 0 AND filtered_signal[i+1] <= 0.
// Uses VPCOMPRESSD (AVX-512VL) to compact matching indices into edges_out in
// ascending order.
void ZeroCrossingDetectAndCollect(const double *filtered_signal, int y_length,
    int *edges_out, int *count_out) {
  const __m512d vzero = _mm512_setzero_pd();
  // Offsets {1, 2, 3, 4, 5, 6, 7, 8} as int32: the stored value is (i+1).
  const __m256i vbase_offset =
      _mm256_setr_epi32(1, 2, 3, 4, 5, 6, 7, 8);

  int count = 0;
  int i = 0;
  const int simd_end = (y_length - 1) & ~7;
  for (; i < simd_end; i += 8) {
    const __m512d xc = _mm512_loadu_pd(filtered_signal + i);
    const __m512d xn = _mm512_loadu_pd(filtered_signal + i + 1);
    const __mmask8 m_pos    = _mm512_cmp_pd_mask(xc, vzero, _CMP_GT_OQ);
    const __mmask8 m_nonpos = _mm512_cmp_pd_mask(xn, vzero, _CMP_LE_OQ);
    const __mmask8 events   = m_pos & m_nonpos;
    if (events) {
      const __m256i idx = _mm256_add_epi32(_mm256_set1_epi32(i), vbase_offset);
      _mm256_mask_compressstoreu_epi32(edges_out + count, events, idx);
      count += _mm_popcnt_u32(events);
    }
  }
  // Scalar tail.
  for (; i < y_length - 1; ++i) {
    if (0.0 < filtered_signal[i] && filtered_signal[i + 1] <= 0.0) {
      edges_out[count++] = i + 1;
    }
  }
  *count_out = count;
}

// K7. Zero-crossing intervals --------------------------------------------
// intervals[i]  = fs / (fine_edges[i+1] - fine_edges[i])
// locations[i]  = (fine_edges[i] + fine_edges[i+1]) / 2 / fs
void ZeroCrossingIntervals(const double *fine_edges, int count, double fs,
    double *intervals, double *locations) {
  if (count < 2) return;
  const __m512d vfs = _mm512_set1_pd(fs);
  const __m512d vhalf_over_fs = _mm512_set1_pd(0.5 / fs);
  int i = 0;
  const int simd_end = (count - 1) & ~7;
  for (; i < simd_end; i += 8) {
    const __m512d cur = _mm512_loadu_pd(fine_edges + i);
    const __m512d nxt = _mm512_loadu_pd(fine_edges + i + 1);
    const __m512d diff = _mm512_sub_pd(nxt, cur);
    const __m512d sum  = _mm512_add_pd(cur, nxt);
    // intervals = fs / diff
    _mm512_storeu_pd(intervals + i, _mm512_div_pd(vfs, diff));
    // locations = (cur + nxt) * 0.5 / fs   (scalar used `/2.0/fs` exactly;
    // precompute 0.5/fs as a constant loses 1 ulp vs (sum/2)/fs but avoids
    // two divides per iter. For bit-identity with scalar, restore below.)
    _mm512_storeu_pd(locations + i, _mm512_mul_pd(sum, vhalf_over_fs));
  }
  // Scalar tail (matches scalar reference).
  for (; i < count - 1; ++i) {
    intervals[i] = fs / (fine_edges[i + 1] - fine_edges[i]);
    locations[i] = (fine_edges[i] + fine_edges[i + 1]) / 2.0 / fs;
  }
}

// K8. Gaussian 1D interior convolution ----------------------------------
// Produces 8 outputs per iter. Inner loop scans kernel[j], broadcasts each
// coefficient, FMAs into 8-wide accumulator with stride-1 loads.
//
// NOTE: The broadcast-kernel-over-i strategy preserves the scalar version's
// summation order (j = 0, 1, 2, ... ascending) for each i, so ULP match is
// preserved up to FMA fusion differences.
void GaussianFilter1DInterior(const double *x, int x_length,
    const double *kernel, int radius, double *y) {
  const int kernel_length = 2 * radius + 1;
  const int interior_begin = radius;
  const int interior_end = x_length - radius;
  int i = interior_begin;
  const int simd_end = interior_end - ((interior_end - interior_begin) & 7);
  for (; i < simd_end; i += 8) {
    __m512d acc = _mm512_setzero_pd();
    const double *xbase = x + (i - radius);
    for (int j = 0; j < kernel_length; ++j) {
      const __m512d kv = _mm512_set1_pd(kernel[j]);
      const __m512d xv = _mm512_loadu_pd(xbase + j);
      acc = _mm512_fmadd_pd(xv, kv, acc);
    }
    _mm512_storeu_pd(y + i, acc);
  }
  // Scalar tail (matches scalar reference: j ascending, plain accumulation).
  for (; i < interior_end; ++i) {
    double val = 0.0;
    for (int j = 0; j < kernel_length; ++j) {
      val += x[i + j - radius] * kernel[j];
    }
    y[i] = val;
  }
}

// K9. Fast Nuttall window via 8-lane parallel recursive oscillators ---------
//
// Scalar reference maintains 3 recursive (c_k, s_k) pairs (k=1..3) advancing
// by phase_step per output sample:
//   y[i] = 0.355768 - 0.487396*c1 + 0.144232*c2 - 0.012604*c3
//
// AVX-512 version maintains 3 *banks* of 8-lane state. Lane j of bank k holds
// (cos(j*k*w), sin(j*k*w)) initially; per iteration the bank advances by
// 8*k*w so the next iteration's lane 0 corresponds to step 8, lane 1 to step
// 9, etc. Each iteration thus produces 8 output samples.
//
// Numerical behavior differs slightly from scalar (different accumulation
// path for the recursion), but drift stays well within the overall pipeline
// tolerance; verified via bench --verify-f0 against scalar golden.
void FastNuttallWindow8(int y_length, double *y) {
  if (y_length <= 1) {
    if (y_length == 1) y[0] = 1.0;
    return;
  }
  // Init cost is ~24 trig calls; below this the scalar loop wins.
  if (y_length < 16) {
    scalar::FastNuttallWindow8(y_length, y);
    return;
  }

  const double w = 2.0 * world::kPi / (y_length - 1);
  const double w1 = w, w2 = 2.0 * w, w3 = 3.0 * w;

  // Advance factors: rotate each bank by 8 steps per iteration.
  const double cos8_1 = cos(8.0 * w1), sin8_1 = sin(8.0 * w1);
  const double cos8_2 = cos(8.0 * w2), sin8_2 = sin(8.0 * w2);
  const double cos8_3 = cos(8.0 * w3), sin8_3 = sin(8.0 * w3);

  // Initial state: lane j = step j.
  alignas(64) double c1_init[8], s1_init[8];
  alignas(64) double c2_init[8], s2_init[8];
  alignas(64) double c3_init[8], s3_init[8];
  for (int j = 0; j < 8; ++j) {
    const double phase1 = j * w1;
    const double phase2 = j * w2;
    const double phase3 = j * w3;
    c1_init[j] = cos(phase1); s1_init[j] = sin(phase1);
    c2_init[j] = cos(phase2); s2_init[j] = sin(phase2);
    c3_init[j] = cos(phase3); s3_init[j] = sin(phase3);
  }
  __m512d c1 = _mm512_load_pd(c1_init);
  __m512d s1 = _mm512_load_pd(s1_init);
  __m512d c2 = _mm512_load_pd(c2_init);
  __m512d s2 = _mm512_load_pd(s2_init);
  __m512d c3 = _mm512_load_pd(c3_init);
  __m512d s3 = _mm512_load_pd(s3_init);

  const __m512d k0 = _mm512_set1_pd(0.355768);
  const __m512d k1 = _mm512_set1_pd(0.487396);
  const __m512d k2 = _mm512_set1_pd(0.144232);
  const __m512d k3 = _mm512_set1_pd(0.012604);
  const __m512d v_cos8_1 = _mm512_set1_pd(cos8_1);
  const __m512d v_sin8_1 = _mm512_set1_pd(sin8_1);
  const __m512d v_cos8_2 = _mm512_set1_pd(cos8_2);
  const __m512d v_sin8_2 = _mm512_set1_pd(sin8_2);
  const __m512d v_cos8_3 = _mm512_set1_pd(cos8_3);
  const __m512d v_sin8_3 = _mm512_set1_pd(sin8_3);

  const int simd_end = y_length & ~7;
  int i = 0;
  for (; i < simd_end; i += 8) {
    __m512d yv = k0;
    yv = _mm512_fnmadd_pd(c1, k1, yv);  // yv -= 0.487396 * c1
    yv = _mm512_fmadd_pd(c2, k2, yv);   // yv += 0.144232 * c2
    yv = _mm512_fnmadd_pd(c3, k3, yv);  // yv -= 0.012604 * c3
    _mm512_storeu_pd(y + i, yv);

    // Rotate each bank by 8*k*w: (c,s) ← (c*cos - s*sin, s*cos + c*sin).
    const __m512d nc1 = _mm512_fmsub_pd(c1, v_cos8_1, _mm512_mul_pd(s1, v_sin8_1));
    const __m512d ns1 = _mm512_fmadd_pd(s1, v_cos8_1, _mm512_mul_pd(c1, v_sin8_1));
    c1 = nc1; s1 = ns1;
    const __m512d nc2 = _mm512_fmsub_pd(c2, v_cos8_2, _mm512_mul_pd(s2, v_sin8_2));
    const __m512d ns2 = _mm512_fmadd_pd(s2, v_cos8_2, _mm512_mul_pd(c2, v_sin8_2));
    c2 = nc2; s2 = ns2;
    const __m512d nc3 = _mm512_fmsub_pd(c3, v_cos8_3, _mm512_mul_pd(s3, v_sin8_3));
    const __m512d ns3 = _mm512_fmadd_pd(s3, v_cos8_3, _mm512_mul_pd(c3, v_sin8_3));
    c3 = nc3; s3 = ns3;
  }

  // Scalar tail: after the loop, lane 0 of each bank holds the state at
  // position simd_end (= advanced by 8 exactly simd_end/8 times).
  if (i < y_length) {
    double c1s = _mm_cvtsd_f64(_mm512_castpd512_pd128(c1));
    double s1s = _mm_cvtsd_f64(_mm512_castpd512_pd128(s1));
    double c2s = _mm_cvtsd_f64(_mm512_castpd512_pd128(c2));
    double s2s = _mm_cvtsd_f64(_mm512_castpd512_pd128(s2));
    double c3s = _mm_cvtsd_f64(_mm512_castpd512_pd128(c3));
    double s3s = _mm_cvtsd_f64(_mm512_castpd512_pd128(s3));
    const double cos1 = cos(w1), sin1 = sin(w1);
    const double cos2 = cos(w2), sin2 = sin(w2);
    const double cos3 = cos(w3), sin3 = sin(w3);
    for (; i < y_length; ++i) {
      y[i] = 0.355768 - 0.487396 * c1s + 0.144232 * c2s - 0.012604 * c3s;
      double nc;
      nc = c1s * cos1 - s1s * sin1; s1s = s1s * cos1 + c1s * sin1; c1s = nc;
      nc = c2s * cos2 - s2s * sin2; s2s = s2s * cos2 + c2s * sin2; c2s = nc;
      nc = c3s * cos3 - s3s * sin3; s3s = s3s * cos3 + c3s * sin3; c3s = nc;
    }
  }
}

// K10. Cosine modulation in place via 8-lane recursive oscillator -----------
//
// Scalar: waveform[i] *= cos(start_phase + i*w_step), via recursion with
// (c, s) advancing by w_step each step.
// AVX-512: 8-lane bank where lane j = (cos(start_phase + j*w_step), sin(...)),
// advances by 8*w_step per iteration.
void CosineModulateInPlace(double *waveform, int length, double start_phase,
    double w_step) {
  if (length <= 0) return;
  if (length < 16) {
    scalar::CosineModulateInPlace(waveform, length, start_phase, w_step);
    return;
  }

  const double cos8 = cos(8.0 * w_step);
  const double sin8 = sin(8.0 * w_step);

  alignas(64) double c_init[8], s_init[8];
  for (int j = 0; j < 8; ++j) {
    const double phase = start_phase + j * w_step;
    c_init[j] = cos(phase);
    s_init[j] = sin(phase);
  }
  __m512d c = _mm512_load_pd(c_init);
  __m512d s = _mm512_load_pd(s_init);

  const __m512d v_cos8 = _mm512_set1_pd(cos8);
  const __m512d v_sin8 = _mm512_set1_pd(sin8);

  const int simd_end = length & ~7;
  int i = 0;
  for (; i < simd_end; i += 8) {
    const __m512d wv = _mm512_loadu_pd(waveform + i);
    _mm512_storeu_pd(waveform + i, _mm512_mul_pd(wv, c));
    const __m512d nc = _mm512_fmsub_pd(c, v_cos8, _mm512_mul_pd(s, v_sin8));
    const __m512d ns = _mm512_fmadd_pd(s, v_cos8, _mm512_mul_pd(c, v_sin8));
    c = nc; s = ns;
  }

  if (i < length) {
    double cs = _mm_cvtsd_f64(_mm512_castpd512_pd128(c));
    double ss = _mm_cvtsd_f64(_mm512_castpd512_pd128(s));
    const double cos_w = cos(w_step);
    const double sin_w = sin(w_step);
    for (; i < length; ++i) {
      waveform[i] *= cs;
      const double nc = cs * cos_w - ss * sin_w;
      ss = ss * cos_w + cs * sin_w;
      cs = nc;
    }
  }
}

}  // namespace avx512
}  // namespace simd
}  // namespace world

#else  // WORLD_HAS_AVX512

// Empty translation unit when compiler lacks AVX-512 support.

#endif  // WORLD_HAS_AVX512
