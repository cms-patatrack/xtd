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

  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t sqrt(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hsqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::sqrt(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::sqrt(static_cast<math_t>(arg));
#endif
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrt(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrtf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrtf(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::sqrt(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::sqrt(arg);
#else
    // standard C/C++ code
    return ::sqrt(arg);
#endif
  }

  /* Computes the square root of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double sqrt(std::integral auto arg) {
    return xtd::sqrt(static_cast<double>(arg));
  }

  /* Computes the square root of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::floating_point auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float sqrtf(std::integral auto arg) {
    return xtd::sqrt(static_cast<float>(arg));
  }

  /* Computes the square root of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t sqrtf16(xtd::floating_point auto arg) {
    return xtd::sqrt(static_cast<float16_t>(arg));
  }

}  // namespace xtd
