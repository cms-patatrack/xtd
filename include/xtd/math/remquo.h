/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"
#include "xtd/math/isfinite.h"

namespace xtd {

  /* Computes the remainder of the floating-point division between the two arguments, in single precision.
   * The returned value is not guaranteed to have the same sign as the first argument.
   * The sign and at least 3 bits of the quotient are stored in the integer value pointed to by the quo pointer.
   *
   * Note: if either argument is infinite or NaN or the denominator is 0, the quotient will be set to 0.
   */
  XTD_DEVICE_FUNCTION inline constexpr float remquo(float x, float y, int *quo) {
    float result;
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    result = ::remquof(x, y, quo);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    result = ::remquof(x, y, quo);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    result = sycl::remquo(x, y, quo);
#else
    // standard C/C++ code
    result = ::remquof(x, y, quo);
#endif
    if (not xtd::isfinite(x) or not xtd::isfinite(y) or y == 0) {
      *quo = 0;
    }
    return result;
  }

  /* Computes the remainder of the floating-point division between the two arguments, in double precision.
   * The returned value is not guaranteed to have the same sign as the first argument.
   * The sign and at least 3 bits of the quotient are stored in the integer value pointed to by the quo pointer.
   *
   * Note: if either argument is infinite or NaN or the denominator is 0, the quotient will be set to 0.
   */
  XTD_DEVICE_FUNCTION inline constexpr double remquo(double x, double y, int *quo) {
    double result;
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    result = ::remquo(x, y, quo);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    result = ::remquo(x, y, quo);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    result = sycl::remquo(x, y, quo);
#else
    // standard C/C++ code
    result = ::remquo(x, y, quo);
#endif
    if (not xtd::isfinite(x) or not xtd::isfinite(y) or y == 0) {
      *quo = 0;
    }
    return result;
  }

  /* Computes the remainder of the floating-point division between the two arguments, in double precision.
   * The returned value is not guaranteed to have the same sign as the first argument.
   * The sign and at least 3 bits of the quotient are stored in the integer value pointed to by the quo pointer.
   *
   * Note: if either argument is infinite or NaN or the denominator is 0, the quotient will be set to 0.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double remquo(Integral x, Integral y, int *quo) {
    return xtd::remquo(static_cast<double>(x), static_cast<double>(y), quo);
  }

  /* Computes the remainder of the floating-point division between the two arguments, in single precision.
   * The returned value is not guaranteed to have the same sign as the first argument.
   * The sign and at least 3 bits of the quotient are stored in the integer value pointed to by the quo pointer.
   *
   * Note: if either argument is infinite or NaN or the denominator is 0, the quotient will be set to 0.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float remquof(FloatingPoint x, FloatingPoint y, int *quo) {
    return xtd::remquo(static_cast<float>(x), static_cast<float>(y), quo);
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float remquof(Integral x, Integral y, int *quo) {
    return xtd::remquo(static_cast<float>(x), static_cast<float>(y), quo);
  }

}  // namespace xtd
