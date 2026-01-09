/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/mpfr.h"

#ifdef mpfr_fmod
#undef mpfr_fmod
#endif

inline float mpfr_fmodf(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  float result;
  mpfr::fmod(static_cast<mpfr_single>(static_cast<float>(x)), static_cast<mpfr_single>(static_cast<float>(y)))
      .conv(result);
  return result;
}

inline double mpfr_fmod(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  double result;
  mpfr::fmod(static_cast<mpfr_double>(static_cast<double>(x)), static_cast<mpfr_double>(static_cast<double>(y)))
      .conv(result);
  return result;
}
