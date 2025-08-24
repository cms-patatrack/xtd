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

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanhf(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanh(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(std::integral auto arg) {
    return xtd::tanh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::floating_point auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::integral auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }

}  // namespace xtd
