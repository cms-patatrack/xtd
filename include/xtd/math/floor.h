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

  /* Computes the largest integral value that is not greater than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float floor(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::floorf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::floorf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::floor(arg);
#else
    // standard C/C++ code
    return ::floorf(arg);
#endif
  }

  /* Computes the largest integral value that is not greater than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double floor(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::floor(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::floor(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::floor(arg);
#else
    // standard C/C++ code
    return ::floor(arg);
#endif
  }

  /* Computes the largest integral value that is not greater than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double floor(std::integral auto arg) {
    return xtd::floor(static_cast<double>(arg));
  }

  /* Computes the largest integral value that is not greater than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float floorf(std::floating_point auto arg) {
    return xtd::floor(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float floorf(std::integral auto arg) {
    return xtd::floor(static_cast<float>(arg));
  }

}  // namespace xtd
