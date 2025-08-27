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

  /* Computes the larger of two point arguments, treating NaNs as missing data, in single precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fmax(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmaxf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmaxf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmax(x, y);
#else
    // standard C/C++ code
    return ::fmaxf(x, y);
#endif
  }

  /* Computes the larger of two point arguments, treating NaNs as missing data, in double precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fmax(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmax(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmax(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmax(x, y);
#else
    // standard C/C++ code
    return ::fmax(x, y);
#endif
  }

  /* Computes the larger of two point arguments, treating NaNs as missing data, in double precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fmax(Integral x, Integral y) {
    return xtd::fmax(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the larger of two point arguments, treating NaNs as missing data, in single precision.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fmaxf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmax(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fmaxf(Integral x, Integral y) {
    return xtd::fmax(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
