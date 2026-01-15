/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#include <bit>
#include <cstdint>

// xtd headers
#include <xtd/concepts/arithmetic.h>

inline float reference_copysignf(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  constexpr uint32_t sign_mask = 0x80000000u;
  return std::bit_cast<float>((std::bit_cast<uint32_t>(static_cast<float>(y)) & sign_mask) |
                              (std::bit_cast<uint32_t>(static_cast<float>(x)) & ~sign_mask));
}

inline double reference_copysign(xtd::arithmetic auto x, xtd::arithmetic auto y) {
  constexpr uint64_t sign_mask = 0x8000000000000000ull;
  return std::bit_cast<double>((std::bit_cast<uint64_t>(static_cast<double>(y)) & sign_mask) |
                               (std::bit_cast<uint64_t>(static_cast<double>(x)) & ~sign_mask));
}
