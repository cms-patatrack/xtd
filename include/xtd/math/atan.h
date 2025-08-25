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

  /* Computes the inverse tangent (measured in radians) or arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float atan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atanf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atanf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C/C++ code
    return ::atanf(arg);
#endif
  }

  /* Computes the inverse tangent (measured in radians) or arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double atan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C/C++ code
    return ::atan(arg);
#endif
  }

  /* Computes the inverse tangent (measured in radians) or arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double atan(std::integral auto arg) {
    return xtd::atan(static_cast<double>(arg));
  }

  /* Computes the inverse tangent (measured in radians) or arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float atanf(std::floating_point auto arg) {
    return xtd::atan(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float atanf(std::integral auto arg) {
    return xtd::atan(static_cast<float>(arg));
  }

}  // namespace xtd
