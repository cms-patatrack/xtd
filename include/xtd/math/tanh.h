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
    // HIP is missing this specific function
    return __float2half(::tanhf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::tanh(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_tanhf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::tanh(static_cast<float>(arg));
    }
    return __builtin_tanhf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_tanhf)
    if (std::is_constant_evaluated()) {
      return std::tanh(static_cast<float>(arg));
    }
    return __builtin_tanhf(static_cast<float>(arg));
#else
    return std::tanh(static_cast<float>(arg));
#endif
#else
    return std::tanh(static_cast<float_t>(arg));
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
  XTD_DEVICE_FUNCTION inline constexpr float16_t tanhf(xtd::floating_point auto arg) {
    return xtd::tanh(static_cast<float16_t>(arg));
  }

}  // namespace xtd
