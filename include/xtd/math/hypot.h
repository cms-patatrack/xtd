/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "xtd/internal/defines.h"

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

  XTD_DEVICE_FUNCTION inline constexpr float hypot(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C++ code
    return std::hypot(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr double hypot(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hypot(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::hypot(x, y);
#else
    // standard C++ code
    return std::hypot(x, y);
#endif
  }

  XTD_DEVICE_FUNCTION inline constexpr float hypotf(float x, float y) { return hypot(x, y); }

}  // namespace xtd
