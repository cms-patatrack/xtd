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

  /* Computes the inverse tangent (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t atan(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::atanf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::atanf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::atan(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::atan(static_cast<math_t>(arg));
#endif
  }

  /* Computes the inverse tangent (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float atan(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atanf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atanf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C/C++ code
    return ::atanf(arg);
#endif
  }

  /* Computes the inverse tangent (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double atan(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::atan(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::atan(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::atan(arg);
#else
    // standard C/C++ code
    return ::atan(arg);
#endif
  }

  /* Computes the inverse tangent (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double atan(std::integral auto arg) {
    return xtd::atan(static_cast<double>(arg));
  }

  /* Computes the inverse tangent (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float atanf(std::floating_point auto arg) {
    return xtd::atan(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float atanf(std::integral auto arg) {
    return xtd::atan(static_cast<float>(arg));
  }

  /* Computes the inverse tangent (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t atanf16(xtd::floating_point auto arg) {
    return xtd::atan(static_cast<float16_t>(arg));
  }

}  // namespace xtd
