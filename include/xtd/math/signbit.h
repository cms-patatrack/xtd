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

  /* Returns one if the single precision floating-point argument is negative, including -0.0f, -inf or a negative NaN.
   * Returns zero otherwise.
   */
  XTD_DEVICE_FUNCTION inline constexpr int signbit(float arg) {
    constexpr uint32_t sign_mask = 0x80000000u;
    uint32_t bits = std::bit_cast<uint32_t>(arg);
    return (bits & sign_mask) ? 1 : 0;
  }

  /* Returns one if the double precision floating-point argument is negative, including -0.0, -inf or a negative NaN.
   * Returns zero otherwise.
   */
  XTD_DEVICE_FUNCTION inline constexpr int signbit(double arg) {
    constexpr uint64_t sign_mask = 0x8000000000000000ull;
    uint64_t bits = std::bit_cast<uint64_t>(arg);
    return (bits & sign_mask) ? 1 : 0;
  }

  /* Returns one if the integer argument is negative, or zero otherwise.
   */
  XTD_DEVICE_FUNCTION inline constexpr int signbit(std::signed_integral auto arg) {
    return arg < 0 ? 1 : 0;
  }

  XTD_DEVICE_FUNCTION inline constexpr int signbit(std::unsigned_integral auto arg) {
    return 0;
  }

}  // namespace xtd
