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

  XTD_DEVICE_FUNCTION inline constexpr float log2(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(x);
#else
    // standard C++ code
    return std::log2(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double log2(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log2(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log2(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log2(x);
#else
    // standard C++ code
    return std::log2(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float log2f(float x) { return log2(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double log2(T x) {
    return log2(static_cast<double>(x));
  }

}  // namespace xtd
