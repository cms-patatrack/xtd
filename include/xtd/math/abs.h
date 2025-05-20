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

  XTD_DEVICE_FUNCTION inline constexpr float abs(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::abs(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::abs(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(x);
#else
    // standard C++ code
    return std::abs(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double abs(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::abs(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::abs(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(x);
#else
    // standard C++ code
    return std::abs(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float fabsf(float x) { return abs(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double fabs(T x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fabs(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fabs(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(x);
#else
    // standard C++ code
    return fabs(x);
#endif
  }

}  // namespace xtd
