/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <limits>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/mpfr.h"

#ifdef mpfr_lrint
#undef mpfr_lrint
#endif

inline long mpfr_lrintf(xtd::arithmetic auto arg) {
  float argf = static_cast<float>(arg);
  if (std::isnan(argf)) {
    // NaN
    return 0;
  } else if (argf >= static_cast<float>(std::numeric_limits<long>::max())) {
    // +infinity and overflow
    return std::numeric_limits<long>::max();
  } else if (argf <= static_cast<float>(std::numeric_limits<long>::min())) {
    // -infinity and underflow
    return std::numeric_limits<long>::min();
  }
  float result;
  mpfr::rint(static_cast<mpfr_single>(argf)).conv(result);
  return static_cast<long>(result);
}

inline long mpfr_lrint(xtd::arithmetic auto arg) {
  double argf = static_cast<double>(arg);
  if (std::isnan(argf)) {
    // NaN
    return 0;
  } else if (argf >= static_cast<double>(std::numeric_limits<long>::max())) {
    // +infinity and overflow
    return std::numeric_limits<long>::max();
  } else if (argf <= static_cast<double>(std::numeric_limits<long>::min())) {
    // -infinity and underflow
    return std::numeric_limits<long>::min();
  }
  double result;
  mpfr::rint(static_cast<mpfr_double>(argf)).conv(result);
  return static_cast<long>(result);
}
