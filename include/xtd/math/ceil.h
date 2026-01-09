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

  /* Computes the smallest integral value that is not less than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float ceil(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceilf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceilf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(arg);
#else
    // standard C/C++ code
    return ::ceilf(arg);
#endif
  }

  /* Computes the smallest integral value that is not less than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double ceil(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::ceil(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::ceil(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::ceil(arg);
#else
    // standard C/C++ code
    return ::ceil(arg);
#endif
  }

  /* Computes the smallest integral value that is not less than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double ceil(std::integral auto arg) {
    return xtd::ceil(static_cast<double>(arg));
  }

  /* Computes the smallest integral value that is not less than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float ceilf(std::floating_point auto arg) {
    return xtd::ceil(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float ceilf(std::integral auto arg) {
    return xtd::ceil(static_cast<float>(arg));
  }

}  // namespace xtd
