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

  /* Computes the inverse cosine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t acos(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::acosf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::acosf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::acos(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::acos(static_cast<math_t>(arg));
#endif
  }

  /* Computes the inverse cosine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C/C++ code
    return ::acosf(arg);
#endif
  }

  /* Computes the inverse cosine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acos(arg);
#else
    // standard C/C++ code
    return ::acos(arg);
#endif
  }

  /* Computes the inverse cosine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acos(std::integral auto arg) {
    return xtd::acos(static_cast<double>(arg));
  }

  /* Computes the inverse cosine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acosf(std::floating_point auto arg) {
    return xtd::acos(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float acosf(std::integral auto arg) {
    return xtd::acos(static_cast<float>(arg));
  }

  /* Computes the inverse cosine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t acosf16(xtd::floating_point auto arg) {
    return xtd::acos(static_cast<float16_t>(arg));
  }

}  // namespace xtd
