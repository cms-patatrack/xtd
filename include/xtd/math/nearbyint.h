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

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float nearbyint(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::nearbyintf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::nearbyintf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rint(arg);
#else
    // standard C/C++ code
    return ::nearbyintf(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr double nearbyint(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::nearbyint(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::nearbyint(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::rint(arg);
#else
    // standard C/C++ code
    return ::nearbyint(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr double nearbyint(std::integral auto arg) {
    return xtd::nearbyint(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases to the nearest even integer value.
   */
  XTD_DEVICE_FUNCTION inline constexpr float nearbyintf(std::floating_point auto arg) {
    return xtd::nearbyint(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float nearbyintf(std::integral auto arg) {
    return xtd::nearbyint(static_cast<float>(arg));
  }

}  // namespace xtd
