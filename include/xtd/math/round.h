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

  /* Computes the nearest integral value to arg in half precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t round(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::roundf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::roundf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::round(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::round(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::round(static_cast<math_t>(arg));
#endif
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float round(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::roundf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::roundf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::round(arg);
#else
    // standard C/C++ code
    return ::roundf(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double round(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::round(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::round(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::round(arg);
#else
    // standard C/C++ code
    return ::round(arg);
#endif
  }

  /* Computes the nearest integral value to arg in double precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr double round(std::integral auto arg) {
    return xtd::round(static_cast<double>(arg));
  }

  /* Computes the nearest integral value to arg in single precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float roundf(std::floating_point auto arg) {
    return xtd::round(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float roundf(std::integral auto arg) {
    return xtd::round(static_cast<float>(arg));
  }

  /* Computes the nearest integral value to arg in half precision, rounding halfway cases away from zero.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t roundf16(xtd::floating_point auto arg) {
    return xtd::round(static_cast<float16_t>(arg));
  }

}  // namespace xtd
