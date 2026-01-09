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

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float round(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::roundf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::roundf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::round(arg);
#else
    // standard C/C++ code
    return ::roundf(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double round(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::round(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::round(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::round(arg);
#else
    // standard C/C++ code
    return ::round(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double round(std::integral auto arg) {
    return xtd::round(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float roundf(std::floating_point auto arg) {
    return xtd::round(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float roundf(std::integral auto arg) {
    return xtd::round(static_cast<float>(arg));
  }

}  // namespace xtd
