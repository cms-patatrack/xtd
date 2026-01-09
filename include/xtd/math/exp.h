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

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::expf(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::exp(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(std::integral auto arg) {
    return xtd::exp(static_cast<double>(arg));
  }

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::floating_point auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::integral auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }

}  // namespace xtd
