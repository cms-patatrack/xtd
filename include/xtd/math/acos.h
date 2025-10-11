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

  /* Computes the inverse cosine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C/C++ code
    return ::acosf(arg);
#endif
  }

  /* Computes the inverse cosine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C/C++ code
    return ::acos(arg);
#endif
  }

  /* Computes the inverse cosine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acos(std::integral auto arg) {
    return xtd::acos(static_cast<double>(arg));
  }

  /* Computes the inverse cosine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acosf(std::floating_point auto arg) {
    return xtd::acos(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float acosf(std::integral auto arg) {
    return xtd::acos(static_cast<float>(arg));
  }

}  // namespace xtd
