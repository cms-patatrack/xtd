/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <bit>
#include <cstdint>
#include <limits>

// xtd headers
#include "xtd/internal/float16.h"

// Properties of the IEEE 754 binary16 format, used by std::numeric_limits<xtd::float16_t>.
#define XTD_FLT16_RADIX 2
#define XTD_FLT16_MANT_DIG 11
#define XTD_FLT16_DIG 3
#define XTD_FLT16_DECIMAL_DIG 5
#define XTD_FLT16_MIN_EXP (-13)
#define XTD_FLT16_MIN_10_EXP (-4)
#define XTD_FLT16_MAX_EXP 16
#define XTD_FLT16_MAX_10_EXP 4
#define XTD_FLT16_HAS_INFINITY 1
#define XTD_FLT16_HAS_QUIET_NAN 1

// Default values. Should be overridden in configuration files if necessary.
#define XTD_FLT16_HAS_DENORM 1
#define XTD_FLT16_HAS_DENORM_LOSS false
#define XTD_FLT16_TRAPS false
#define XTD_FLT16_TINYNESS_BEFORE false

namespace std {
  /// numeric_limits<xtd::float16_t> specialization.
  template<>
    struct numeric_limits<xtd::float16_t>
    {
      static constexpr bool is_specialized = true;
      static constexpr xtd::float16_t min() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x0400)); }
      static constexpr xtd::float16_t max() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7bff)); }
      static constexpr xtd::float16_t lowest() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0xfbff)); }

      static constexpr int digits = XTD_FLT16_MANT_DIG;
      static constexpr int digits10 = XTD_FLT16_DIG;
      static constexpr int max_digits10 = XTD_FLT16_DECIMAL_DIG;
      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = XTD_FLT16_RADIX;

      static constexpr xtd::float16_t
      epsilon() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x1400)); }

      static constexpr xtd::float16_t round_error() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x3800)); }

      static constexpr int min_exponent = XTD_FLT16_MIN_EXP;
      static constexpr int min_exponent10 = XTD_FLT16_MIN_10_EXP;
      static constexpr int max_exponent = XTD_FLT16_MAX_EXP;
      static constexpr int max_exponent10 = XTD_FLT16_MAX_10_EXP;

      static constexpr bool has_infinity = XTD_FLT16_HAS_INFINITY;
      static constexpr bool has_quiet_NaN = XTD_FLT16_HAS_QUIET_NAN;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm = bool(XTD_FLT16_HAS_DENORM) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss = XTD_FLT16_HAS_DENORM_LOSS;

      static constexpr xtd::float16_t infinity() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7C00)); }
      static constexpr xtd::float16_t quiet_NaN() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7E00)); }
      static constexpr xtd::float16_t signaling_NaN() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7D00)); }
      static constexpr xtd::float16_t denorm_min() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x0001)); }

      static constexpr bool is_iec559 = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = XTD_FLT16_TRAPS;
      static constexpr bool tinyness_before = XTD_FLT16_TINYNESS_BEFORE;
      static constexpr float_round_style round_style = round_to_nearest;
    };
} // namespace std
