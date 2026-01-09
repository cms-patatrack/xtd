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

  /* Computes the natural logarithm of (1 + arg), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log1p(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log1pf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log1pf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log1p(arg);
#else
    // standard C/C++ code
    return ::log1pf(arg);
#endif
  }

  /* Computes the natural logarithm of (1 + arg), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log1p(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log1p(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log1p(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log1p(arg);
#else
    // standard C/C++ code
    return ::log1p(arg);
#endif
  }

  /* Computes the natural logarithm of (1 + arg), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log1p(std::integral auto arg) {
    return xtd::log1p(static_cast<double>(arg));
  }

  /* Computes the natural logarithm of (1 + arg), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log1pf(std::floating_point auto arg) {
    return xtd::log1p(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log1pf(std::integral auto arg) {
    return xtd::log1p(static_cast<float>(arg));
  }

}  // namespace xtd
