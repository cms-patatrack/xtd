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

  /* Computes the inverse hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acosh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acoshf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acoshf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C/C++ code
    return ::acoshf(arg);
#endif
  }

  /* Computes the inverse hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acosh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C/C++ code
    return ::acosh(arg);
#endif
  }

  /* Computes the inverse hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acosh(std::integral auto arg) {
    return xtd::acosh(static_cast<double>(arg));
  }

  /* Computes the inverse hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acoshf(std::floating_point auto arg) {
    return xtd::acosh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float acoshf(std::integral auto arg) {
    return xtd::acosh(static_cast<float>(arg));
  }

}  // namespace xtd
