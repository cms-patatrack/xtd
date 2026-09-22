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

  /* Computes the smaller of the two arguments, in half precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t fmin(float16_t x, float16_t y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::__hmin(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::__hmin(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmin(static_cast<sycl::half>(x), static_cast<sycl::half>(y));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T a, T b) { { std::fmin(a, b) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, double>;
#else
    using math_t = double;
#endif
    return std::fmin(static_cast<math_t>(x), static_cast<math_t>(y));
#endif
  }

  /* Computes the smaller of the two arguments, in single precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fmin(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fminf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fminf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmin(x, y);
#else
    // standard C/C++ code
    return ::fminf(x, y);
#endif
  }

  /* Computes the smaller of the two arguments, in double precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fmin(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmin(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmin(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmin(x, y);
#else
    // standard C/C++ code
    return ::fmin(x, y);
#endif
  }

  /* Computes the smaller of the two arguments, in double precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fmin(Integral x, Integral y) {
    return xtd::fmin(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the smaller of the two arguments, in single precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fminf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmin(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fminf(Integral x, Integral y) {
    return xtd::fmin(static_cast<float>(x), static_cast<float>(y));
  }

  /* Computes the smaller of the two arguments, in half precision, treating NaNs as missing data.
   * Between a NaN and a numeric value, the numeric value is chosen.
   */
  template <xtd::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float16_t fminf16(FloatingPoint x, FloatingPoint y) {
    return xtd::fmin(static_cast<float16_t>(x), static_cast<float16_t>(y));
  }

}  // namespace xtd
