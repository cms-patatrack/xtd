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

  /* Computes the cosine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cosf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cosf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C/C++ code
    return ::cosf(arg);
#endif
  }

  /* Computes the cosine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C/C++ code
    return ::cos(arg);
#endif
  }

  /* Computes the cosine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cos(std::integral auto arg) {
    return xtd::cos(static_cast<double>(arg));
  }

  /* Computes the cosine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cosf(std::floating_point auto arg) {
    return xtd::cos(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float cosf(std::integral auto arg) {
    return xtd::cos(static_cast<float>(arg));
  }

}  // namespace xtd
