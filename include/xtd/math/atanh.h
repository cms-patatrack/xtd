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
  inline constexpr float atanh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atanh(arg);
#else
    // standard C++ code
    return std::atanh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double atanh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atanh(arg);
#else
    // standard C++ code
    return std::atanh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float atanhf(float arg) { return atanh(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double atanh(T arg) {
    return atanh(static_cast<double>(arg));
  }

}  // namespace xtd
