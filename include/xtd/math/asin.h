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

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asinf(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asin(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(std::integral auto arg) {
    return xtd::asin(static_cast<double>(arg));
  }

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::floating_point auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::integral auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }

}  // namespace xtd
