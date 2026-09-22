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

  /* Computes the largest integral value that is not greater than arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t floor(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::hfloor(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::hfloor(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::floor(static_cast<sycl::half>(arg));
#else
    // standard C/C++ code
#if defined(XTD_HAS_STDFLOAT16)
    constexpr bool native = []<typename T>(T) { return requires(T v) { { std::floor(v) } -> std::same_as<T>; }; }(_Float16{});
    using math_t = std::conditional_t<native, _Float16, float>;
#else
    using math_t = float;
#endif
    return std::floor(static_cast<math_t>(arg));
#endif
  }

  /* Computes the largest integral value that is not greater than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float floor(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::floorf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::floorf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::floor(arg);
#else
    // standard C/C++ code
    return ::floorf(arg);
#endif
  }

  /* Computes the largest integral value that is not greater than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double floor(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::floor(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::floor(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::floor(arg);
#else
    // standard C/C++ code
    return ::floor(arg);
#endif
  }

  /* Computes the largest integral value that is not greater than arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double floor(std::integral auto arg) {
    return xtd::floor(static_cast<double>(arg));
  }

  /* Computes the largest integral value that is not greater than arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float floorf(std::floating_point auto arg) {
    return xtd::floor(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float floorf(std::integral auto arg) {
    return xtd::floor(static_cast<float>(arg));
  }

  /* Computes the largest integral value that is not greater than arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t floorf16(xtd::floating_point auto arg) {
    return xtd::floor(static_cast<float16_t>(arg));
  }

}  // namespace xtd
