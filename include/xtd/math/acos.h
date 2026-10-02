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
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_acosf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::acos(static_cast<float>(arg));
    }
    return __builtin_acosf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_acosf)
    if (std::is_constant_evaluated()) {
      return std::acos(static_cast<float>(arg));
    }
    return __builtin_acosf(static_cast<float>(arg));
#else
    return std::acos(static_cast<float>(arg));
#endif
#else
    return std::acos(static_cast<float_t>(arg));
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
  XTD_DEVICE_FUNCTION inline constexpr float16_t acosf(xtd::floating_point auto arg) {
    return xtd::acos(static_cast<float16_t>(arg));
  }

}  // namespace xtd
