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

  /* Computes the smaller of two point arguments, treating NaNs as missing data, in single precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fmin(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fminf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fminf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmin(x, y);
#else
    // standard C/C++ code
    return ::fminf(x, y);
#endif
  }

  /* Computes the smaller of two point arguments, treating NaNs as missing data, in double precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fmin(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmin(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmin(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmin(x, y);
#else
    // standard C/C++ code
    return ::fmin(x, y);
#endif
  }

  /* Computes the smaller of two point arguments, treating NaNs as missing data, in double precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fmin(Integral x, Integral y) {
    return xtd::fmin(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the smaller of two point arguments, treating NaNs as missing data, in single precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fminf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmin(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fminf(Integral x, Integral y) {
    return xtd::fmin(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
