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

#ifdef mpfr_pow
#undef mpfr_pow
#endif

inline float mpfr_powf(xtd::arithmetic auto base, xtd::arithmetic auto exp) {
  float result;
  mpfr::pow(static_cast<mpfr_single>(static_cast<float>(base)), static_cast<mpfr_single>(static_cast<float>(exp)))
      .conv(result);
  return result;
}

inline double mpfr_pow(xtd::arithmetic auto base, xtd::arithmetic auto exp) {
  double result;
  mpfr::pow(static_cast<mpfr_double>(static_cast<double>(base)), static_cast<mpfr_double>(static_cast<double>(exp)))
      .conv(result);
  return result;
}
