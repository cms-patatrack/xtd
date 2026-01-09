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

  /* Computes the hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cosh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::coshf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::coshf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(arg);
#else
    // standard C/C++ code
    return ::coshf(arg);
#endif
  }

  /* Computes the hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cosh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cosh(arg);
#else
    // standard C/C++ code
    return ::cosh(arg);
#endif
  }

  /* Computes the hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cosh(std::integral auto arg) {
    return xtd::cosh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float coshf(std::floating_point auto arg) {
    return xtd::cosh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float coshf(std::integral auto arg) {
    return xtd::cosh(static_cast<float>(arg));
  }

}  // namespace xtd
