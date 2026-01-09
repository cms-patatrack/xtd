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

#ifdef mpfr_atan2
#undef mpfr_atan2
#endif

inline float mpfr_atan2f(xtd::arithmetic auto y, xtd::arithmetic auto x) {
  float result;
  mpfr::atan2(static_cast<mpfr_single>(static_cast<float>(y)), static_cast<mpfr_single>(static_cast<float>(x)))
      .conv(result);
  return result;
}

inline double mpfr_atan2(xtd::arithmetic auto y, xtd::arithmetic auto x) {
  double result;
  mpfr::atan2(static_cast<mpfr_double>(static_cast<double>(y)), static_cast<mpfr_double>(static_cast<double>(x)))
      .conv(result);
  return result;
}
