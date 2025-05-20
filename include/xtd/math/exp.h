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

  XTD_DEVICE_FUNCTION inline constexpr float exp(float x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(x);
#else
    // standard C++ code
    return std::exp(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double exp(double x) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp(x);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp(x);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(x);
#else
    // standard C++ code
    return std::exp(x);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float expf(float x) { return exp(x); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double exp(T x) {
    return exp(static_cast<double>(x));
  }

}  // namespace xtd
