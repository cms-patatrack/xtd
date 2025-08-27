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

  /* Computes the positive difference between the two arguments, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fdim(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fdimf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fdimf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fdim(x, y);
#else
    // standard C/C++ code
    return ::fdimf(x, y);
#endif
  }

  /* Computes the positive difference between the two arguments, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fdim(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fdim(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fdim(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fdim(x, y);
#else
    // standard C/C++ code
    return ::fdim(x, y);
#endif
  }

  /* Computes the positive difference between the two arguments, in double precision.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fdim(Integral x, Integral y) {
    return xtd::fdim(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the positive difference between the two arguments, in single precision.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fdimf(FloatingPoint x, FloatingPoint y) {
    return xtd::fdim(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fdimf(Integral x, Integral y) {
    return xtd::fdim(static_cast<float>(x), static_cast<float>(y));
  }

}  // namespace xtd
