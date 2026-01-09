/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <bit>
#include <cmath>
#include <cstdint>

#include "xtd/internal/defines.h"

namespace xtd {

  /* Returns a single precision quiet NaN (Not a Number) value.
   *
   * Note: the lower bits of the argument are encoded in the lower bits on the NaN.
   */
  XTD_DEVICE_FUNCTION inline constexpr float nanf(uint32_t arg = 0u) {
    return std::bit_cast<float>(std::bit_cast<uint32_t>(static_cast<float>(NAN)) | arg);
  }

  /* Returns a double precision quiet NaN (Not a Number) value.
   *
   * Note: the lower bits of the argument are encoded in the lower bits on the NaN.
   */
  XTD_DEVICE_FUNCTION inline constexpr double nan(uint64_t arg = 0ull) {
    return std::bit_cast<double>(std::bit_cast<uint64_t>(static_cast<double>(NAN)) | arg);
  }

}  // namespace xtd
