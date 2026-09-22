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

  /* Computes the absolute value of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t fabs(float16_t arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return __habs(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return __habs(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(static_cast<sycl::half>(arg));
#elif XTD_HAS_STDFLOAT16
    // standard C/C++ code
    return ::fabs(std::bit_cast<std::float16_t>(arg));
#else
    // standard C/C++ code
    return arg & uint16_t{0x7FFF};
#endif
  }

  /* Computes the absolute value of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fabs(float arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fabsf(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fabsf(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(arg);
#else
    // standard C/C++ code
    return ::fabsf(arg);
#endif
  }

  /* Computes the absolute value of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fabs(double arg) {
#if defined(XTD_TARGET_CUDA)
    // CUDA device code
    return ::fabs(arg);
#elif defined(XTD_TARGET_HIP)
    // HIP/ROCm device code
    return ::fabs(arg);
#elif defined(XTD_TARGET_SYCL)
    // SYCL device code
    return sycl::fabs(arg);
#else
    // standard C/C++ code
    return ::fabs(arg);
#endif
  }

  /* Computes the absolute value of arg, in double precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr double fabs(std::integral auto arg) {
    return xtd::fabs(static_cast<double>(arg));
  }

  /* Computes the absolute value of arg, in single precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float fabsf(std::floating_point auto arg) {
    return xtd::fabs(static_cast<float>(arg));
  }
  XTD_DEVICE_FUNCTION inline constexpr float fabsf(std::integral auto arg) {
    return xtd::fabs(static_cast<float>(arg));
  }

  /* Computes the absolute value of arg, in half precision.
   */
  XTD_DEVICE_FUNCTION inline constexpr float16_t fabsf(xtd::floating_point auto arg) {
    return xtd::fabs(static_cast<float16_t>(arg));
  }

}  // namespace xtd
