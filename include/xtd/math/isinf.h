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

  /* Returns +1 or -1 if the single precision argument is positive or negative infinity, or zero otherwise.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isinf(float arg) {
    uint32_t bits = std::bit_cast<uint32_t>(arg);
    constexpr uint32_t exp_mask = 0x7f800000u;
    constexpr uint32_t mant_mask = 0x007fffffu;
    constexpr uint32_t sign_mask = 0x80000000u;
    if ((bits & exp_mask) == exp_mask and (bits & mant_mask) == 0)
      return (bits & sign_mask) ? -1 : +1;
    return 0;
  }

  /* Returns +1 or -1 if the double precision argument is positive or negative infinity, or zero otherwise.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isinf(double arg) {
    uint64_t bits = std::bit_cast<uint64_t>(arg);
    constexpr uint64_t exp_mask = 0x7ff0000000000000ull;
    constexpr uint64_t mant_mask = 0x000fffffffffffffull;
    constexpr uint64_t sign_mask = 0x8000000000000000ull;
    if ((bits & exp_mask) == exp_mask and (bits & mant_mask) == 0)
      return (bits & sign_mask) ? -1 : +1;
    return 0;
  }

  /* Returns a non-zero value if the integer argument is infinite, which is never the case.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isinf(std::integral auto arg) {
    return 0;
  }

}  // namespace xtd
