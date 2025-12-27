/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <bit>
#include <concepts>
#include <cstdint>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Returns a non-zero value if the single precision argument is finite (normal, subnormal, or zero).
   */
  XTD_DEVICE_FUNCTION inline constexpr int isfinite(float arg) {
    uint32_t bits = std::bit_cast<uint32_t>(arg);
    constexpr uint32_t mask = 0x7f800000u;
    uint32_t exp = bits & mask;
    return exp != mask;
  }

  /* Returns a non-zero value if the double precision argument is finite (normal, subnormal, or zero).
   */
  XTD_DEVICE_FUNCTION inline constexpr int isfinite(double arg) {
    uint64_t bits = std::bit_cast<uint64_t>(arg);
    constexpr uint64_t mask = 0x7ff0000000000000ull;
    uint64_t exp = bits & mask;
    return exp != mask;
  }

  /* Returns a non-zero value if the integer argument is finite (which is always the case).
   */
  XTD_DEVICE_FUNCTION inline constexpr int isfinite(std::integral auto arg) {
    return 1;
  }

}  // namespace xtd
