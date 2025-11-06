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

#ifdef mpfr_log10
#undef mpfr_log10
#endif

inline float mpfr_log10f(xtd::arithmetic auto arg) {
  float result;
  mpfr::log10(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_log10(xtd::arithmetic auto arg) {
  double result;
  mpfr::log10(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
