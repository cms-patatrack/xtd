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

  /* Computes e raised to the given power, minus 1, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t expm1(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::expm1f(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::expm1f(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::expm1(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::expm1(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::expm1(static_cast<math_t>(arg));
#endif
  }

  /* Computes e raised to the given power, minus 1, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expm1(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expm1f(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expm1f(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::expm1(arg);
#else
    // standard C/C++ code
    return ::expm1f(arg);
#endif
  }

  /* Computes e raised to the given power, minus 1, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double expm1(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expm1(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expm1(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::expm1(arg);
#else
    // standard C/C++ code
    return ::expm1(arg);
#endif
  }

  /* Computes e raised to the given power, minus 1, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double expm1(std::integral auto arg) {
    return xtd::expm1(static_cast<double>(arg));
  }

  /* Computes e raised to the given power, minus 1, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expm1f(std::floating_point auto arg) {
    return xtd::expm1(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float expm1f(std::integral auto arg) {
    return xtd::expm1(static_cast<float>(arg));
  }

  /* Computes the base-e exponential minus one of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t expm1f16(xtd::floating_point auto arg) {
    return xtd::expm1(static_cast<float16_t>(arg));
  }

}  // namespace xtd
