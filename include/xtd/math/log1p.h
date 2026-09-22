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

  /* Computes the natural logarithm of (1 + arg), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log1p(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::log1pf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::log1pf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log1p(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::log1p(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::log1p(static_cast<math_t>(arg));
#endif
  }

  /* Computes the natural logarithm of (1 + arg), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log1p(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log1pf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log1pf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log1p(arg);
#else
    // standard C/C++ code
    return ::log1pf(arg);
#endif
  }

  /* Computes the natural logarithm of (1 + arg), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log1p(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::log1p(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::log1p(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::log1p(arg);
#else
    // standard C/C++ code
    return ::log1p(arg);
#endif
  }

  /* Computes the natural logarithm of (1 + arg), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double log1p(std::integral auto arg) {
    return xtd::log1p(static_cast<double>(arg));
  }

  /* Computes the natural logarithm of (1 + arg), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float log1pf(std::floating_point auto arg) {
    return xtd::log1p(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float log1pf(std::integral auto arg) {
    return xtd::log1p(static_cast<float>(arg));
  }

  /* Computes the natural logarithm of (1 + arg), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t log1pf16(xtd::floating_point auto arg) {
    return xtd::log1p(static_cast<float16_t>(arg));
  }

}  // namespace xtd
