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

  /* Computes the reciprocal of the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float rsqrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    // Note: __frsqrt_rn() is correctly rounded, while rsqrtf() is rounded to 2 ULPs
    return ::__frsqrt_rn(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::rsqrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rsqrt(arg);
#else
    // standard C/C++ code
    return 1.f / ::sqrtf(arg);
#endif
  }

  /* Computes the reciprocal of the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double rsqrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::rsqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::rsqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rsqrt(arg);
#else
    // standard C/C++ code
    return 1. / ::sqrt(arg);
#endif
  }

  /* Computes the reciprocal of the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double rsqrt(std::integral auto arg) {
    return xtd::rsqrt(static_cast<double>(arg));
  }

  /* Computes the reciprocal of the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float rsqrtf(std::floating_point auto arg) {
    return xtd::rsqrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float rsqrtf(std::integral auto arg) {
    return xtd::rsqrt(static_cast<float>(arg));
  }

}  // namespace xtd
