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

#ifdef mpfr_floor
#undef mpfr_floor
#endif

inline float mpfr_floorf(xtd::arithmetic auto arg) {
  float result;
  mpfr::floor(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_floor(xtd::arithmetic auto arg) {
  double result;
  mpfr::floor(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
