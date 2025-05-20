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
  inline constexpr float acosh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C++ code
    return std::acosh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double acosh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C++ code
    return std::acosh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float acoshf(float arg) { return acosh(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double acosh(T arg) {
    return acosh(static_cast<double>(arg));
  }

}  // namespace xtd
