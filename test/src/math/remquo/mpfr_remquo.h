/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++
#include <cmath>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/mpfr.h"

#ifdef mpfr_remquo
#undef mpfr_remquo
#endif

inline float mpfr_remquof(xtd::arithmetic auto x, xtd::arithmetic auto y, int* q) {
  float result = 0.f;
  long int quotient = 0;
  mpfr::remquo(
      static_cast<mpfr_single>(static_cast<float>(x)), static_cast<mpfr_single>(static_cast<float>(y)), &quotient)
      .conv(result);
  // low part of the quotient, including the sign and at least 7 bits (ISO C and C++ require 3 bits, OpenCL and SYCL provide 7 bits)
  if (std::isfinite(x) and std::isfinite(y) and y != 0) {
    quotient %= 256;
    if (std::signbit(x) xor std::signbit(y)) {
      if (quotient > 0)
        quotient -= 256;
    } else {
      if (quotient < 0)
        quotient += 256;
    }
    *q = static_cast<int>(quotient);
  } else {
    *q = 0;
  }

  return result;
}

inline double mpfr_remquo(xtd::arithmetic auto x, xtd::arithmetic auto y, int* q) {
  double result = 0.;
  long int quotient = 0;
  mpfr::remquo(
      static_cast<mpfr_double>(static_cast<double>(x)), static_cast<mpfr_double>(static_cast<double>(y)), &quotient)
      .conv(result);
  // low part of the quotient, including the sign and at least 7 bits (ISO C and C++ require 3 bits, OpenCL and SYCL provide 7 bits)
  if (std::isfinite(x) and std::isfinite(y) and y != 0) {
    quotient %= 256;
    if (std::signbit(x) xor std::signbit(y)) {
      if (quotient > 0)
        quotient -= 256;
    } else {
      if (quotient < 0)
        quotient += 256;
    }
    *q = static_cast<int>(quotient);
  } else {
    *q = 0;
  }

  return result;
}
