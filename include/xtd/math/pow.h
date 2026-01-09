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

  /* Computes the value of base raised to the power exp, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float pow(float base, float exp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::powf(base, exp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::powf(base, exp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::pow(base, exp);
#else
    // standard C/C++ code
    return ::powf(base, exp);
#endif
  }

  /* Computes the value of base raised to the power exp, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double pow(double base, double exp) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::pow(base, exp);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::pow(base, exp);
#else
    // standard C/C++ code
    return ::pow(base, exp);
#endif
  }

  /* Computes the value of base raised to the power exp, in double precision.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double pow(Integral base, Integral exp) {
    return xtd::pow(static_cast<double>(base), static_cast<double>(exp));
  }

  /* Computes the value of base raised to the power exp, in single precision.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float powf(FloatingPoint base, FloatingPoint exp) {
    return xtd::pow(static_cast<float>(base), static_cast<float>(exp));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float powf(Integral base, Integral exp) {
    return xtd::pow(static_cast<float>(base), static_cast<float>(exp));
  }

}  // namespace xtd
