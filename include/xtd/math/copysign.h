/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Returns a single precision value with the absolute value of x and the sign of y.
   */
  XTD_DEVICE_FUNCTION inline constexpr float copysign(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::copysignf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::copysignf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::copysign(x, y);
#else
    // standard C/C++ code
    return ::copysignf(x, y);
#endif
  }

  /* Returns a double precision value with the absolute value of x and the sign of y.
   */
  XTD_DEVICE_FUNCTION inline constexpr double copysign(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::copysign(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::copysign(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::copysign(x, y);
#else
    // standard C/C++ code
    return ::copysign(x, y);
#endif
  }

  /* Returns a double precision value with the absolute value of x and the sign of y.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double copysign(Integral x, Integral y) {
    return xtd::copysign(static_cast<double>(x), static_cast<double>(y));
  }

  /* Returns a single precision value with the absolute value of x and the sign of y.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float copysignf(FloatingPoint x, FloatingPoint y) {
    return xtd::copysign(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float copysignf(Integral x, Integral y) {
    return xtd::copysign(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
