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

#ifdef mpfr_rint
#undef mpfr_rint
#endif

inline xtd::float16_t mpfr_rintf16(xtd::float16_t x) {
  return test_half::reference(
      [](mpfr_ptr r, mpfr_srcptr a, mpfr_rnd_t rnd) { return ::mpfr_rint(r, a, rnd); }, x);
}

inline float mpfr_rintf(xtd::arithmetic auto arg) {
  float result;
  mpfr::rint(static_cast<mpfr_single>(static_cast<float>(arg))).conv(result);
  return result;
}

inline double mpfr_rint(xtd::arithmetic auto arg) {
  double result;
  mpfr::rint(static_cast<mpfr_double>(static_cast<double>(arg))).conv(result);
  return result;
}
