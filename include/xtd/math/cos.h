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

  /* Computes the cosine of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cos(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hcos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hcos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::cos(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::cos(static_cast<math_t>(arg));
#endif
  }

  /* Computes the cosine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cos(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cosf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cosf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C/C++ code
    return ::cosf(arg);
#endif
  }

  /* Computes the cosine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cos(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::cos(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::cos(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::cos(arg);
#else
    // standard C/C++ code
    return ::cos(arg);
#endif
  }

  /* Computes the cosine of arg (measured in radians), in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double cos(std::integral auto arg) {
    return xtd::cos(static_cast<double>(arg));
  }

  /* Computes the cosine of arg (measured in radians), in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float cosf(std::floating_point auto arg) {
    return xtd::cos(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float cosf(std::integral auto arg) {
    return xtd::cos(static_cast<float>(arg));
  }

  /* Computes the cosine of arg (measured in radians), in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t cosf16(xtd::floating_point auto arg) {
    return xtd::cos(static_cast<float16_t>(arg));
  }

}  // namespace xtd
