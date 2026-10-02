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

  /* Computes the inverse sine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t asin(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __float2half(::asinf(__half2float(arg)));
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __float2half(::asinf(__half2float(arg)));
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_asinf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::asin(static_cast<float>(arg));
    }
    return __builtin_asinf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_asinf)
    if (std::is_constant_evaluated()) {
      return std::asin(static_cast<float>(arg));
    }
    return __builtin_asinf(static_cast<float>(arg));
#else
    return std::asin(static_cast<float>(arg));
#endif
#else
    return std::asin(static_cast<float_t>(arg));
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asin(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asinf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asinf(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::asin(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::asin(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::asin(arg);
#else
    // standard C/C++ code
    return ::asin(arg);
#endif
  }

  /* Computes the inverse sine (measured in radians) of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double asin(std::integral auto arg) {
    return xtd::asin(static_cast<double>(arg));
  }

  /* Computes the inverse sine (measured in radians) of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::floating_point auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float asinf(std::integral auto arg) {
    return xtd::asin(static_cast<float>(arg));
  }

  /* Computes the inverse sine (measured in radians) of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t asinf(xtd::floating_point auto arg) {
    return xtd::asin(static_cast<float16_t>(arg));
  }

}  // namespace xtd
