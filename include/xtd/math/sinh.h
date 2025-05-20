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
  inline constexpr float sinh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C++ code
    return std::sinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double sinh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sinh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sinh(arg);
#else
    // standard C++ code
    return std::sinh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float sinhf(float arg) { return sinh(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double sinh(T arg) {
    return sinh(static_cast<double>(arg));
  }

}  // namespace xtd
