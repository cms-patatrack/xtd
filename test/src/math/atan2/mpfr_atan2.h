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
// #include "common/math_inputs.h"
#include "common/mpfr.h"

#ifdef mpfr_atan2
#undef mpfr_atan2
#endif

// inline xtd::float16_t mpfr_atan2f16(xtd::float16_t x, xtd::float16_t y) {
//   return test_half::reference(
//       [](mpfr_ptr r, mpfr_srcptr a, mpfr_srcptr b, mpfr_rnd_t rnd) { return ::mpfr_atan2(r, a, b, rnd); }, x, y);
// }

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
