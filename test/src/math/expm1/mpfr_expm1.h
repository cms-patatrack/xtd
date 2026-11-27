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

#ifdef mpfr_expm1
#undef mpfr_expm1
#endif

inline xtd::float16_t mpfr_expm1f16(xtd::float16_t x) {
  return test_half::reference(
      [](mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd) { return ::mpfr_expm1(r, a, rnd); }, x);
}

inline float mpfr_expm1f(xtd::arithmetic auto arg) {
  float result;
  mpfr::expm1(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_expm1(xtd::arithmetic auto arg) {
  double result;
  mpfr::expm1(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
