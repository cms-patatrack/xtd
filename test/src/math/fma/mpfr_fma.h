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

#ifdef mpfr_fma
#undef mpfr_fma
#endif

inline float mpfr_fmaf(xtd::arithmetic auto x, xtd::arithmetic auto y, xtd::arithmetic auto z) {
  float result;
  mpfr::fma(static_cast<mpfr_single>(static_cast<float>(x)),
            static_cast<mpfr_single>(static_cast<float>(y)),
            static_cast<mpfr_single>(static_cast<float>(z)))
      .conv(result);
  return result;
}

inline double mpfr_fma(xtd::arithmetic auto x, xtd::arithmetic auto y, xtd::arithmetic auto z) {
  double result;
  mpfr::fma(static_cast<mpfr_double>(static_cast<double>(x)),
            static_cast<mpfr_double>(static_cast<double>(y)),
            static_cast<mpfr_double>(static_cast<double>(z)))
      .conv(result);
  return result;
}
