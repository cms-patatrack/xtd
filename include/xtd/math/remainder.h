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

  /* Computes the IEEE remainder of the floating-point division the two arguments, in single precision.
   * In contrast to xtd::fmod, the returned value is not guaranteed to have the same sign as x.
   */
  XTD_DEVICE_FUNCTION inline constexpr float remainder(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::remainderf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::remainderf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::remainder(x, y);
#else
    // standard C/C++ code
    return ::remainderf(x, y);
#endif
  }

  /* Computes the IEEE remainder of the floating-point division the two arguments, in double precision.
   * In contrast to xtd::fmod, the returned value is not guaranteed to have the same sign as x.
   */
  XTD_DEVICE_FUNCTION inline constexpr double remainder(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::remainder(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::remainder(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::remainder(x, y);
#else
    // standard C/C++ code
    return ::remainder(x, y);
#endif
  }

  /* Computes the IEEE remainder of the floating-point division the two arguments, in double precision.
   * In contrast to xtd::fmod, the returned value is not guaranteed to have the same sign as x.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double remainder(Integral x, Integral y) {
    return xtd::remainder(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the IEEE remainder of the floating-point division the two arguments, in single precision.
   * In contrast to xtd::fmod, the returned value is not guaranteed to have the same sign as x.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float remainderf(FloatingPoint x, FloatingPoint y) {
    return xtd::remainder(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float remainderf(Integral x, Integral y) {
    return xtd::remainder(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
