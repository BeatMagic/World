//-----------------------------------------------------------------------------
// Copyright 2026 justln1113
//
// CompositeF0 combines DIO and Harvest F0 estimators with zero-crossing rate
// analysis for more robust voiced/unvoiced decision.
//-----------------------------------------------------------------------------
#ifndef WORLD_COMPOSITEF0_H_
#define WORLD_COMPOSITEF0_H_

#include "world/macrodefinitions.h"

WORLD_BEGIN_C_DECLS

//-----------------------------------------------------------------------------
// Struct for CompositeF0
//-----------------------------------------------------------------------------
typedef struct {
  double f0_floor;
  double f0_ceil;
  double frame_period;      // msec
  int zcr_frame_length;     // samples for ZCR computation
  double zcr_threshold;     // threshold for ZCR derivative
  double gaussian_sigma;    // sigma for Gaussian smoothing of ZCR derivative
} CompositeF0Option;

//-----------------------------------------------------------------------------
// CompositeF0
//
// Input:
//   x                    : Input signal
//   x_length             : Length of x
//   fs                   : Sampling frequency
//   option               : Struct to order the parameter for CompositeF0
//
// Output:
//   temporal_positions   : Temporal positions.
//   f0                   : F0 contour.
//-----------------------------------------------------------------------------
void CompositeF0(const double *x, int x_length, int fs,
  const CompositeF0Option *option, double *temporal_positions, double *f0);

//-----------------------------------------------------------------------------
// InitializeCompositeF0Option allocates the memory to the struct and sets the
// default parameters.
//
// Output:
//   option   : Struct for the optional parameter.
//-----------------------------------------------------------------------------
void InitializeCompositeF0Option(CompositeF0Option *option);

//-----------------------------------------------------------------------------
// GetSamplesForCompositeF0() calculates the number of samples required for
// CompositeF0().
//
// Input:
//   fs             : Sampling frequency [Hz]
//   x_length       : Length of the input signal [Sample]
//   frame_period   : Frame shift [msec]
//
// Output:
//   The number of samples required to store the results of CompositeF0()
//-----------------------------------------------------------------------------
int GetSamplesForCompositeF0(int fs, int x_length, double frame_period);

WORLD_END_C_DECLS

#endif  // WORLD_COMPOSITEF0_H_
