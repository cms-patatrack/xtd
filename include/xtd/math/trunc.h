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

  /* Computes the nearest integral value to arg in single precision, always rounding towards zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float trunc(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::truncf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::truncf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::trunc(arg);
#else
    // standard C/C++ code
    return ::truncf(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, always rounding towards zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double trunc(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::trunc(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::trunc(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::trunc(arg);
#else
    // standard C/C++ code
    return ::trunc(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, always rounding towards zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double trunc(std::integral auto arg) {
    return xtd::trunc(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg in single precision, always rounding towards zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float truncf(std::floating_point auto arg) {
    return xtd::trunc(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float truncf(std::integral auto arg) {
    return xtd::trunc(static_cast<float>(arg));
  }

}  // namespace xtd
