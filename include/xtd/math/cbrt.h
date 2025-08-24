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

  /* Computes the cubic root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrtf(arg);
#endif
  }

  /* Computes the cubic root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrt(arg);
#endif
  }

  /* Computes the cubic root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(std::integral auto arg) {
    return xtd::cbrt(static_cast<double>(arg));
  }

  /* Computes the cubic root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::floating_point auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::integral auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }

}  // namespace xtd
