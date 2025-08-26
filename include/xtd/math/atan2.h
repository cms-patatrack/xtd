/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Computes the inverse tangent (measured in radians) or y/x, in single precision,
   * using the signs of the two arguments to determine the quadrant of the result.
   */
  XTD_DEVICE_FUNCTION inline constexpr float atan2(float y, float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan2f(y, x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan2f(y, x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan2(y, x);
#else
    // standard C/C++ code
    return ::atan2f(y, x);
#endif
  }

  /* Computes the inverse tangent (measured in radians) or y/x, in double precision,
   * using the signs of the two arguments to determine the quadrant of the result.
   */
  XTD_DEVICE_FUNCTION inline constexpr double atan2(double y, double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan2(y, x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan2(y, x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan2(y, x);
#else
    // standard C/C++ code
    return ::atan2(y, x);
#endif
  }

  /* Computes the inverse tangent (measured in radians) or y/x, in double precision,
   * using the signs of the two arguments to determine the quadrant of the result.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double atan2(Integral y, Integral x) {
    return xtd::atan2(static_cast<double>(y), static_cast<double>(x));
  }

  /* Computes the inverse tangent (measured in radians) or y/x, in single precision,
   * using the signs of the two arguments to determine the quadrant of the result.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float atan2f(FloatingPoint y, FloatingPoint x) {
    return xtd::atan2(static_cast<float>(y), static_cast<float>(x));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float atan2f(Integral y, Integral x) {
    return xtd::atan2(static_cast<float>(y), static_cast<float>(x));
  }

}  // namespace xtd
