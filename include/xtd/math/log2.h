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

  /* Computes the base-2 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log2(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(arg);
#else
    // standard C/C++ code
    return ::log2f(arg);
#endif
  }

  /* Computes the base-2 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log2(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(arg);
#else
    // standard C/C++ code
    return ::log2(arg);
#endif
  }

  /* Computes the base-2 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log2(std::integral auto arg) {
    return xtd::log2(static_cast<double>(arg));
  }

  /* Computes the base-2 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log2f(std::floating_point auto arg) {
    return xtd::log2(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log2f(std::integral auto arg) {
    return xtd::log2(static_cast<float>(arg));
  }

}  // namespace xtd
