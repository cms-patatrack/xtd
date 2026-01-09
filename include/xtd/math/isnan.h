/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <bit>
#include <concepts>
#include <cstdint>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Returns a non-zero value if the single precision argument is "Not a Number".
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnan(float arg) {
    uint32_t bits = std::bit_cast<uint32_t>(arg);
    constexpr uint32_t exp_mask = 0x7f800000u;
    constexpr uint32_t mant_mask = 0x007fffffu;
    return (bits & exp_mask) == exp_mask and (bits & mant_mask) != 0;
  }

  /* Returns a non-zero value if the double precision argument is "Not a Number".
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnan(double arg) {
    uint64_t bits = std::bit_cast<uint64_t>(arg);
    constexpr uint64_t exp_mask = 0x7ff0000000000000ull;
    constexpr uint64_t mant_mask = 0x000fffffffffffffull;
    return (bits & exp_mask) == exp_mask and (bits & mant_mask) != 0;
  }

  /* Returns a non-zero value if the integer argument is "Not a Number", which
   * is never the case.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnan(std::integral auto arg) {
    return 0;
  }

}  // namespace xtd
