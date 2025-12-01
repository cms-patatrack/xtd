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

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lround(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::lroundf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::lroundf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return static_cast<long>(sycl::round(arg));
#else
    // standard C/C++ code
    return ::lroundf(arg);
#endif
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lround(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::lround(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::lround(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return static_cast<long>(sycl::round(arg));
#else
    // standard C/C++ code
    return ::lround(arg);
#endif
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lround(std::integral auto arg) {
    return xtd::lround(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg as a signed long int, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr long lroundf(std::floating_point auto arg) {
    return xtd::lround(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr long lroundf(std::integral auto arg) {
    return xtd::lround(static_cast<float>(arg));
  }

}  // namespace xtd
