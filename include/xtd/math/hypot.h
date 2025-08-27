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

  /* Computes the square root of the sum of the squares of the two arguments, in single precision,
   * without undue overflow or underflow at intermediate stages of the computation.
   */
  XTD_DEVICE_FUNCTION inline constexpr float hypot(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypotf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypotf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C/C++ code
    return ::hypotf(x, y);
#endif
  }

  /* Computes the square root of the sum of the squares of the two arguments, in double precision,
   * without undue overflow or underflow at intermediate stages of the computation.
   */
  XTD_DEVICE_FUNCTION inline constexpr double hypot(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C/C++ code
    return ::hypot(x, y);
#endif
  }

  /* Computes the square root of the sum of the squares of the two arguments, in double precision,
   * without undue overflow or underflow at intermediate stages of the computation.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double hypot(Integral x, Integral y) {
    return xtd::hypot(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the square root of the sum of the squares of the two arguments, in single precision,
   * without undue overflow or underflow at intermediate stages of the computation.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float hypotf(FloatingPoint x, FloatingPoint y) {
    return xtd::hypot(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float hypotf(Integral x, Integral y) {
    return xtd::hypot(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
