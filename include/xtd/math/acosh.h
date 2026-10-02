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
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_acoshf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::acosh(static_cast<float>(arg));
    }
    return __builtin_acoshf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_acoshf)
    if (std::is_constant_evaluated()) {
      return std::acosh(static_cast<float>(arg));
    }
    return __builtin_acoshf(static_cast<float>(arg));
#else
    return std::acosh(static_cast<float>(arg));
#endif
#else
    return std::acosh(static_cast<float_t>(arg));
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

}  // namespace xtd
