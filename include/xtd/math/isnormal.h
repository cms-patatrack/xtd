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

  /* Returns a non-zero value if the single precision argument is normal.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnormal(float arg) {
    uint32_t bits = std::bit_cast<uint32_t>(arg);
    constexpr uint32_t mask = 0x7f800000u;
    uint32_t exp = bits & mask;
    return exp != 0 and exp != mask;
  }

  /* Returns a non-zero value if the double precision argument is normal.
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnormal(double arg) {
    uint64_t bits = std::bit_cast<uint64_t>(arg);
    constexpr uint64_t mask = 0x7ff0000000000000ull;
    uint64_t exp = bits & mask;
    return exp != 0 and exp != mask;
  }

  /* Returns a non-zero value if the integer argument is normal.
   * Zero is arbitrarily considered "not normal" for consistency with
   * isnormal(0.f) and isnormal(0.)
   */
  XTD_DEVICE_FUNCTION inline constexpr int isnormal(std::integral auto arg) {
    return arg != 0;
  }

}  // namespace xtd
