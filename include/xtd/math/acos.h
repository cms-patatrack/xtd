/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "xtd/internal/defines.h"
#include <concepts>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION
  inline constexpr float acos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C++ code
    return std::acos(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double acos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C++ code
    return std::acos(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float acosf(float arg) { return acos(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double acos(T arg) {
    return acos(static_cast<double>(arg));
  }

}  // namespace xtd
