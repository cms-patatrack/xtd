/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Computes the value of x apply.sh doc function_arg.h function_arg_arg.h function_arg_arg_arg.h functions patch test_arg test_arg_arg test_arg_arg_arg y + z as if calculated to infinite precision and rounded once to single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fma(float x, float y, float z) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmaf(x, y, z);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmaf(x, y, z);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fma(x, y, z);
#else
    // standard C/C++ code
    return ::fmaf(x, y, z);
#endif
  }

  /* Computes the value of x apply.sh doc function_arg.h function_arg_arg.h function_arg_arg_arg.h functions patch test_arg test_arg_arg test_arg_arg_arg y + z as if calculated to infinite precision and rounded once to double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fma(double x, double y, double z) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fma(x, y, z);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fma(x, y, z);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fma(x, y, z);
#else
    // standard C/C++ code
    return ::fma(x, y, z);
#endif
  }

  /* Computes the value of x apply.sh doc function_arg.h function_arg_arg.h function_arg_arg_arg.h functions patch test_arg test_arg_arg test_arg_arg_arg y + z as if calculated to infinite precision and rounded once to double precision.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fma(Integral x, Integral y, Integral z) {
    return xtd::fma(static_cast<double>(x), static_cast<double>(y), static_cast<double>(z));
  }

  /* Computes the value of x apply.sh doc function_arg.h function_arg_arg.h function_arg_arg_arg.h functions patch test_arg test_arg_arg test_arg_arg_arg y + z as if calculated to infinite precision and rounded once to single precision.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fmaf(FloatingPoint x, FloatingPoint y, FloatingPoint z) {
    return xtd::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fmaf(Integral x, Integral y, Integral z) {
    return xtd::fma(static_cast<float>(x), static_cast<float>(y), static_cast<float>(z));
  }

}  // namespace xtd
