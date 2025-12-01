/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
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

#ifdef mpfr_llround
#undef mpfr_llround
#endif

inline long long mpfr_llroundf(xtd::arithmetic auto arg) {
  float argf = static_cast<float>(arg);
  if (std::isnan(argf)) {
    // NaN
    return 0;
  } else if (argf >= static_cast<float>(std::numeric_limits<long long>::max())) {
    // +infinity and overflow
    return std::numeric_limits<long long>::max();
  } else if (argf <= static_cast<float>(std::numeric_limits<long long>::min())) {
    // -infinity and underflow
    return std::numeric_limits<long long>::min();
  }
  float result;
  mpfr::round(static_cast<mpfr_single>(argf)).conv(result);
  return static_cast<long long>(result);
}

inline long long mpfr_llround(xtd::arithmetic auto arg) {
  double argf = static_cast<double>(arg);
  if (std::isnan(argf)) {
    // NaN
    return 0;
  } else if (argf >= static_cast<double>(std::numeric_limits<long long>::max())) {
    // +infinity and overflow
    return std::numeric_limits<long long>::max();
  } else if (argf <= static_cast<double>(std::numeric_limits<long long>::min())) {
    // -infinity and underflow
    return std::numeric_limits<long long>::min();
  }
  double result;
  mpfr::round(static_cast<mpfr_double>(argf)).conv(result);
  return static_cast<long long>(result);
}
