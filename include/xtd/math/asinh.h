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

  /* Computes the inverse hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C/C++ code
    return ::asinhf(arg);
#endif
  }

  /* Computes the inverse hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C/C++ code
    return ::asinh(arg);
#endif
  }

  /* Computes the inverse hyperbolic sine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asinh(std::integral auto arg) {
    return xtd::asinh(static_cast<double>(arg));
  }

  /* Computes the inverse hyperbolic sine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinhf(std::floating_point auto arg) {
    return xtd::asinh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float asinhf(std::integral auto arg) {
    return xtd::asinh(static_cast<float>(arg));
  }

}  // namespace xtd
