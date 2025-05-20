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
  inline constexpr float asinh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C++ code
    return std::asinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double asinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asinh(arg);
#else
    // standard C++ code
    return std::asinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float asinhf(float arg) { return asinh(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double asinhf(T arg) {
    return asinh(static_cast<double>(arg));
  }

}  // namespace xtd
