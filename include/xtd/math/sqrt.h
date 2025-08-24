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

  /* Computes the square root, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrtf(arg);
#endif
  }

  /* Computes the square root, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrt(arg);
#endif
  }

  /* Computes the square root, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(std::integral auto arg) {
    return xtd::sqrt(static_cast<double>(arg));
  }

  /* Computes the square root, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::floating_point auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::integral auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }

}  // namespace xtd
