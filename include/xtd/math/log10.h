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

  /* Computes the base 10 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log10(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log10f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log10f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log10(arg);
#else
    // standard C/C++ code
    return ::log10f(arg);
#endif
  }

  /* Computes the base 10 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log10(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log10(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log10(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log10(arg);
#else
    // standard C/C++ code
    return ::log10(arg);
#endif
  }

  /* Computes the base 10 logarithm of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log10(std::integral auto arg) {
    return xtd::log10(static_cast<double>(arg));
  }

  /* Computes the base 10 logarithm of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log10f(std::floating_point auto arg) {
    return xtd::log10(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log10f(std::integral auto arg) {
    return xtd::log10(static_cast<float>(arg));
  }

}  // namespace xtd
