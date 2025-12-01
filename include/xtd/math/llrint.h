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

  /* Computes the nearest integral value to arg as a signed long long int, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long long llrint(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::llrintf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::llrintf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return static_cast<long long>(sycl::rint(arg));
#else
    // standard C/C++ code
    return ::llrintf(arg);
#endif
  }

  /* Computes the nearest integral value to arg as a signed long long int, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long long llrint(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::llrint(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::llrint(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return static_cast<long long>(sycl::rint(arg));
#else
    // standard C/C++ code
    return ::llrint(arg);
#endif
  }

  /* Computes the nearest integral value to arg as a signed long long int, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long long llrint(std::integral auto arg) {
    return xtd::llrint(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg as a signed long long int, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr long long llrintf(std::floating_point auto arg) {
    return xtd::llrint(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr long long llrintf(std::integral auto arg) {
    return xtd::llrint(static_cast<float>(arg));
  }

}  // namespace xtd
