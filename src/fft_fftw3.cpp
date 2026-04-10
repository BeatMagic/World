//-----------------------------------------------------------------------------
// FFTW3 backend for World's FFT interface.
// FFTW3 provides heavily optimized FFT with AVX2/AVX-512 SIMD.
//
// Build: make FFT_BACKEND=fftw3  (requires libfftw3-dev)
//
// Key difference from previous attempt: plan pre-caching in
// RefinementWorkspace means FFTW plans are created only ~6 times
// (once per fft_size), not thousands of times. This eliminates
// the plan creation overhead that made FFTW3 3x slower before.
//-----------------------------------------------------------------------------
#include "world/fft.h"

#include <string.h>
#include <fftw3.h>

fft_plan fft_plan_dft_1d(int n, fft_complex *in, fft_complex *out, int sign,
    unsigned int flags) {
  fft_plan output = {0};
  output.n = n;
  output.in = NULL;
  output.c_in = in;
  output.out = NULL;
  output.c_out = out;
  output.sign = sign;
  output.flags = flags;
  output.ip = NULL;
  output.w = NULL;

  int fftw_sign = (sign == FFT_FORWARD) ? FFTW_FORWARD : FFTW_BACKWARD;
  fftw_plan p = fftw_plan_dft_1d(n,
      reinterpret_cast<fftw_complex *>(in),
      reinterpret_cast<fftw_complex *>(out),
      fftw_sign, FFTW_ESTIMATE);
  output.input = reinterpret_cast<double *>(p);
  return output;
}

fft_plan fft_plan_dft_c2r_1d(int n, fft_complex *in, double *out,
    unsigned int flags) {
  fft_plan output = {0};
  output.n = n;
  output.in = NULL;
  output.c_in = in;
  output.out = out;
  output.c_out = NULL;
  output.sign = FFT_BACKWARD;
  output.flags = flags;
  output.ip = NULL;
  output.w = NULL;

  fftw_plan p = fftw_plan_dft_c2r_1d(n,
      reinterpret_cast<fftw_complex *>(in), out, FFTW_ESTIMATE);
  output.input = reinterpret_cast<double *>(p);
  return output;
}

fft_plan fft_plan_dft_r2c_1d(int n, double *in, fft_complex *out,
    unsigned int flags) {
  fft_plan output = {0};
  output.n = n;
  output.in = in;
  output.c_in = NULL;
  output.out = NULL;
  output.c_out = out;
  output.sign = FFT_FORWARD;
  output.flags = flags;
  output.ip = NULL;
  output.w = NULL;

  fftw_plan p = fftw_plan_dft_r2c_1d(n, in,
      reinterpret_cast<fftw_complex *>(out), FFTW_ESTIMATE);
  output.input = reinterpret_cast<double *>(p);
  return output;
}

void fft_execute(fft_plan p) {
  fftw_plan plan = reinterpret_cast<fftw_plan>(p.input);
  // Use FFTW "new-array execute" API so that changing p.in/p.c_in/p.c_out/p.out
  // after plan creation takes effect. World code (e.g. DIO) sometimes modifies
  // the fft_plan struct's buffer pointers between plan creation and execution.
  if (p.sign == FFT_FORWARD) {
    if (p.c_in == NULL) {  // r2c
      fftw_execute_dft_r2c(plan, p.in,
          reinterpret_cast<fftw_complex *>(p.c_out));
    } else {  // c2c forward
      fftw_execute_dft(plan,
          reinterpret_cast<fftw_complex *>(p.c_in),
          reinterpret_cast<fftw_complex *>(p.c_out));
    }
  } else {  // FFT_BACKWARD
    if (p.c_out == NULL) {  // c2r
      fftw_execute_dft_c2r(plan,
          reinterpret_cast<fftw_complex *>(p.c_in), p.out);
    } else {  // c2c backward
      fftw_execute_dft(plan,
          reinterpret_cast<fftw_complex *>(p.c_in),
          reinterpret_cast<fftw_complex *>(p.c_out));
    }
  }
}

void fft_destroy_plan(fft_plan p) {
  fftw_plan plan = reinterpret_cast<fftw_plan>(p.input);
  fftw_destroy_plan(plan);
}
