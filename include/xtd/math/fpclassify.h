/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <bit>
#include <cmath>
#include <concepts>
#include <cstdint>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Classify the single precision floating point argument in one of the categories: zero, subnormal, normal, infinite, or "Not a Number".
   */
  XTD_DEVICE_FUNCTION inline constexpr int fpclassify(float arg) {
    constexpr uint32_t exp_mask = 0x7f800000u;
    constexpr uint32_t mant_mask = 0x007fffffu;

    uint32_t bits = std::bit_cast<uint32_t>(arg);
    uint32_t exp = bits & exp_mask;
    uint32_t mant = bits & mant_mask;

    if (exp == exp_mask) {
      return (mant == 0) ? FP_INFINITE : FP_NAN;
    }
    if (exp == 0) {
      return (mant == 0) ? FP_ZERO : FP_SUBNORMAL;
    }
    return FP_NORMAL;
  }

  /* Classify the double precision floating point argument in one of the categories: zero, subnormal, normal, infinite, or "Not a Number".
   */
  XTD_DEVICE_FUNCTION inline constexpr int fpclassify(double arg) {
    constexpr uint64_t exp_mask = 0x7ff0000000000000ull;
    constexpr uint64_t mant_mask = 0x000fffffffffffffull;

    uint64_t bits = std::bit_cast<uint64_t>(arg);
    uint64_t exp = bits & exp_mask;
    uint64_t mant = bits & mant_mask;

    if (exp == exp_mask) {
      return (mant == 0) ? FP_INFINITE : FP_NAN;
    }

    if (exp == 0) {
      return (mant == 0) ? FP_ZERO : FP_SUBNORMAL;
    }

    return FP_NORMAL;
  }

  /* Classify the integer argument in one of the categories: zero, or normal.
   */
  XTD_DEVICE_FUNCTION inline constexpr int fpclassify(std::integral auto arg) {
    return (arg == 0) ? FP_ZERO : FP_NORMAL;
  }

}  // namespace xtd
