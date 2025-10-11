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

  /* Computes 2 raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp2(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp2f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp2f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp2(arg);
#else
    // standard C/C++ code
    return ::exp2f(arg);
#endif
  }

  /* Computes 2 raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp2(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp2(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp2(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp2(arg);
#else
    // standard C/C++ code
    return ::exp2(arg);
#endif
  }

  /* Computes 2 raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp2(std::integral auto arg) {
    return xtd::exp2(static_cast<double>(arg));
  }

  /* Computes 2 raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp2f(std::floating_point auto arg) {
    return xtd::exp2(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float exp2f(std::integral auto arg) {
    return xtd::exp2(static_cast<float>(arg));
  }

}  // namespace xtd
