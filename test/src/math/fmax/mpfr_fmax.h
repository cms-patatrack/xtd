/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/mpfr.h"

#ifdef mpfr_fmax
#undef mpfr_fmax
#endif

inline float mpfr_fmaxf(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  float result;
  mpfr::fmax(static_cast<mpfr_single>(static_cast<float>(x)), static_cast<mpfr_single>(static_cast<float>(y)))
      .conv(result);
  return result;
}

inline double mpfr_fmax(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  double result;
  mpfr::fmax(static_cast<mpfr_double>(static_cast<double>(x)), static_cast<mpfr_double>(static_cast<double>(y)))
      .conv(result);
  return result;
}
