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
  inline constexpr float tanh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // stanhdard C++ code
    return std::tanh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr double tanh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // stanhdard C++ code
    return std::tanh(arg);
#endif
  }

  XTD_DEVICE_FUNCTION
  inline constexpr float tanhf(float arg) { return tanh(arg); }

  template <std::integral T>
  XTD_DEVICE_FUNCTION inline constexpr double tanh(T arg) {
    return tanh(static_cast<double>(arg));
  }

}  // namespace xtd
