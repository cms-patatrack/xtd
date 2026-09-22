/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include "xtd/internal/concepts.h"
#include "xtd/internal/defines.h"

namespace xtd {

  /* Computes the larger of the two arguments, in half precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t fmax(float16_t x, float16_t y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::__hmax(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::__hmax(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmax(static_cast<sycl::half>(x), static_cast<sycl::half>(y));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_fmax16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::fmax(static_cast<float>(x), static_cast<float>(y));
    }
    return __builtin_fmax16(std::bit_cast<std::float16_t>(x), std::bit_cast<std::float16_t>(y));
#elif __has_builtin(__builtin_fmaxf)
    if (std::is_constant_evaluated()) {
      return std::fmax(static_cast<float>(x), static_cast<float>(y));
    }
    return __builtin_fmaxf(static_cast<float>(x), static_cast<float>(y));
#else
    return std::fmax(static_cast<float>(x), static_cast<float>(y));
#endif
#else
    return std::fmax(static_cast<float_t>(x), static_cast<float_t>(y));
#endif
  }

  /* Computes the larger of the two arguments, in single precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fmax(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmaxf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmaxf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmax(x, y);
#else
    // standard C/C++ code
    return ::fmaxf(x, y);
#endif
  }

  /* Computes the larger of the two arguments, in double precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fmax(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmax(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmax(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmax(x, y);
#else
    // standard C/C++ code
    return ::fmax(x, y);
#endif
  }

  /* Computes the larger of the two arguments, in double precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fmax(Integral x, Integral y) {
    return xtd::fmax(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the larger of the two arguments, in single precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fmaxf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmax(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fmaxf(Integral x, Integral y) {
    return xtd::fmax(static_cast<float>(x), static_cast<float>(y));
  }

  /* Computes the larger of the two arguments, in half precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <xtd::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float16_t fmaxf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmax(static_cast<float16_t>(x), static_cast<float16_t>(y));
  }

}  // namespace xtd
