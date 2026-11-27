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

  /* Computes the cube root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cbrt(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::cbrtf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::cbrtf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(static_cast<float>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::cbrt(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::cbrt(static_cast<math_t>(arg));
#endif
  }

  /* Computes the cube root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrtf(arg);
#endif
  }

  /* Computes the cube root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cbrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cbrt(arg);
#else
    // standard C/C++ code
    return ::cbrt(arg);
#endif
  }

  /* Computes the cube root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cbrt(std::integral auto arg) {
    return xtd::cbrt(static_cast<double>(arg));
  }

  /* Computes the cube root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::floating_point auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float cbrtf(std::integral auto arg) {
    return xtd::cbrt(static_cast<float>(arg));
  }

  /* Computes the cube root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cbrtf16(xtd::floating_point auto arg) {
    return xtd::cbrt(static_cast<float16_t>(arg));
  }

}  // namespace xtd
