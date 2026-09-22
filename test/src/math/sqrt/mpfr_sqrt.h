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
#include "common/math_inputs.h"
#include "common/mpfr.h"

#ifdef mpfr_sqrt
#undef mpfr_sqrt
#endif

inline xtd::float16_t mpfr_sqrtf16(xtd::float16_t x) {
  return test_half::reference(
      [](mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd) { return ::mpfr_sqrt(r, a, rnd); }, x);
}

inline float mpfr_sqrtf(xtd::arithmetic auto arg) {
  float result;
  mpfr::sqrt(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_sqrt(xtd::arithmetic auto arg) {
  double result;
  mpfr::sqrt(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
