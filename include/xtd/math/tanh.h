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

  /* Computes the hyperbolic tangent of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t tanh(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::htanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::tanhf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::tanh(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::tanh(static_cast<math_t>(arg));
#endif
  }

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanh(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanhf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanhf(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::tanh(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(arg);
#else
    // standard C/C++ code
    return ::tanh(arg);
#endif
  }

  /* Computes the hyperbolic tangent of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double tanh(std::integral auto arg) {
    return xtd::tanh(static_cast<double>(arg));
  }

  /* Computes the hyperbolic tangent of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::floating_point auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float tanhf(std::integral auto arg) {
    return xtd::tanh(static_cast<float>(arg));
  }

  /* Computes the hyperbolic tangent of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t tanhf16(xtd::floating_point auto arg) {
    return xtd::tanh(static_cast<float16_t>(arg));
  }

}  // namespace xtd
