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

  /* Computes the inverse hyperbolic cosine of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t acosh(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::acoshf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half_rn(::acoshf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::acosh(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::acosh(static_cast<math_t>(arg));
#endif
  }

  /* Computes the inverse hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acosh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acoshf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acoshf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C/C++ code
    return ::acoshf(arg);
#endif
  }

  /* Computes the inverse hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acosh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::acosh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::acosh(arg);
#else
    // standard C/C++ code
    return ::acosh(arg);
#endif
  }

  /* Computes the inverse hyperbolic cosine of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double acosh(std::integral auto arg) {
    return xtd::acosh(static_cast<double>(arg));
  }

  /* Computes the inverse hyperbolic cosine of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float acoshf(std::floating_point auto arg) {
    return xtd::acosh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float acoshf(std::integral auto arg) {
    return xtd::acosh(static_cast<float>(arg));
  }

  /* Computes the inverse hyperbolic cosine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t acoshf16(xtd::floating_point auto arg) {
    return xtd::acosh(static_cast<float16_t>(arg));
  }

}  // namespace xtd
