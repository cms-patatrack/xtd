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

  /* Computes the tangent of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C/C++ code
    return ::tanf(arg);
#endif
  }

  /* Computes the tangent of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tan(arg);
#else
    // standard C/C++ code
    return ::tan(arg);
#endif
  }

  /* Computes the tangent of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tan(std::integral auto arg) {
    return xtd::tan(static_cast<double>(arg));
  }

  /* Computes the tangent of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanf(std::floating_point auto arg) {
    return xtd::tan(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float tanf(std::integral auto arg) {
    return xtd::tan(static_cast<float>(arg));
  }

}  // namespace xtd
