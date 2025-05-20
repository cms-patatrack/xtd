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
  inline constexpr float atan2(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan2(x, y);
#else
    // standard C++ code
    return std::atan2(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double atan2(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan2(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan2(x, y);
#else
    // standard C++ code
    return std::atan2(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float atan2f(float x, float y) { return atan2(x, y); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double atan2(T x, T y) {
    return atan2(static_cast<double>(x), static_cast<double>(y));
  }

}  // namespace xtd
