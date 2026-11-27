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

  /* Computes the floating-point remainder of the division between the two arguments, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t fmod(float16_t x, float16_t y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmodf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmodf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmod(static_cast<sycl::half>(x), static_cast<sycl::half>(y));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T a, T b) { { std::fmod(a, b) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, double>;
#else
    using math_t = double;
#endif
    return std::fmod(static_cast<math_t>(x), static_cast<math_t>(y));
#endif
  }

  /* Computes the floating-point remainder of the division between the two arguments, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fmod(float x, float y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmodf(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmodf(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmod(x, y);
#else
    // standard C/C++ code
    return ::fmodf(x, y);
#endif
  }

  /* Computes the floating-point remainder of the division between the two arguments, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fmod(double x, double y) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fmod(x, y);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fmod(x, y);
#else
    // standard C/C++ code
    return ::fmod(x, y);
#endif
  }

  /* Computes the floating-point remainder of the division between the two arguments, in double precision.
   */
  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr double fmod(Integral x, Integral y) {
    return xtd::fmod(static_cast<double>(x), static_cast<double>(y));
  }

  /* Computes the floating-point remainder of the division between the two arguments, in single precision.
   */
  template <std::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float fmodf(FloatingPoint x, FloatingPoint y) {
    return xtd::fmod(static_cast<float>(x), static_cast<float>(y));
  }

  template <std::integral Integral>
  XTD_DEVICE_FUNCTION inline constexpr float fmodf(Integral x, Integral y) {
    return xtd::fmod(static_cast<float>(x), static_cast<float>(y));
  }

  /* Computes the floating-point remainder of the division between the two arguments, in half precision.
   */
  template <xtd::floating_point FloatingPoint>
  XTD_DEVICE_FUNCTION inline constexpr float16_t fmodf16(FloatingPoint x, FloatingPoint y) {
    return xtd::fmod(static_cast<float16_t>(x), static_cast<float16_t>(y));
  }

}  // namespace xtd
