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

  /* Computes he exponential value of arg, minus 1, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expm1(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expm1f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expm1f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::expm1(arg);
#else
    // standard C/C++ code
    return ::expm1f(arg);
#endif
  }

  /* Computes he exponential value of arg, minus 1, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double expm1(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expm1(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expm1(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::expm1(arg);
#else
    // standard C/C++ code
    return ::expm1(arg);
#endif
  }

  /* Computes he exponential value of arg, minus 1, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double expm1(std::integral auto arg) {
    return xtd::expm1(static_cast<double>(arg));
  }

  /* Computes he exponential value of arg, minus 1, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expm1f(std::floating_point auto arg) {
    return xtd::expm1(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float expm1f(std::integral auto arg) {
    return xtd::expm1(static_cast<float>(arg));
  }

}  // namespace xtd
