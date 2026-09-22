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
  /* Computes e raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t exp(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hexp(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hexp(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
#if defined(__clang__) && __has_builtin(__builtin_expf16) && defined(__AVX512FP16__)
    if (std::is_constant_evaluated()) {
      return std::exp(static_cast<float>(arg));
    }
    return __builtin_expf16(std::bit_cast<std::float16_t>(arg));
#elif __has_builtin(__builtin_expf)
    if (std::is_constant_evaluated()) {
      return std::exp(static_cast<float>(arg));
    }
    return __builtin_expf(static_cast<float>(arg));
#else
    return std::exp(static_cast<float>(arg));
#endif
#else
    return std::exp(static_cast<float_t>(arg));
#endif
  }

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float exp(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::expf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::expf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::expf(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::exp(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::exp(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::exp(arg);
#else
    // standard C/C++ code
    return ::exp(arg);
#endif
  }

  /* Computes e raised to the given power, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double exp(std::integral auto arg) {
    return xtd::exp(static_cast<double>(arg));
  }

  /* Computes e raised to the given power, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::floating_point auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float expf(std::integral auto arg) {
    return xtd::exp(static_cast<float>(arg));
  }

  /* Computes e raised to the given power, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t expf(xtd::floating_point auto arg) {
    return xtd::exp(static_cast<float16_t>(arg));
  }

}  // namespace xtd
