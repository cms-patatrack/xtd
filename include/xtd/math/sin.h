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

  /* Computes the sine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C++ code
    return std::sin(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sin(arg);
#else
    // standard C++ code
    return std::sin(arg);
#endif
  }

  /* Computes the sine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sin(std::integral auto arg) {
    return sin(static_cast<double>(arg));
  }

  /* Computes the sine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sinf(std::floating_point auto arg) { return xtd::sin(static_cast<float>(arg)); }
  XTD_DEVICE_FUNCTION inline constexpr float sinf(std::integral auto arg) { return xtd::sin(static_cast<float>(arg)); }

}  // namespace xtd
