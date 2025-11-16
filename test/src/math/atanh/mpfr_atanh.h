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

#ifdef mpfr_atanh
#undef mpfr_atanh
#endif

inline float mpfr_atanhf(xtd::arithmetic auto arg) {
  float result;
  mpfr::atanh(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_atanh(xtd::arithmetic auto arg) {
  double result;
  mpfr::atanh(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
