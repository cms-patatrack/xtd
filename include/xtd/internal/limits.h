/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <cmath>
#include <limits>

// XTD concepts
#include "xtd/internal/concepts.h"
#include "xtd/internal/float16.h"

// Mirror std::limits
#if !defined(__FLT16_DIG__)
#define __FLT16_HAS_QUIET_NAN__ 1
#define __FLT16_MIN_EXP__ (-13)
#define __FLT16_MAX_10_EXP__ 4
#define __FLT16_DECIMAL_DIG__ 5
#define __FLT16_DENORM_MIN__ 5.960464477539062500000000000e-8f
#define __FLT16_MIN_10_EXP__ (-4)
#define __FLT16_IS_IEC_60559__ 1
#define __FLT16_MIN__ 6.103515625000000000000000000e-5f
#define __FLT16_MAX_EXP__ 16
#define __FLT16_EPSILON__ 9.765625000000000000000000000e-4f
#define __FLT16_NORM_MAX__ 6.550400000000000000000000000e+4f
#define __FLT16_MAX__ 6.550400000000000000000000000e+4f
#define __FLT16_HAS_INFINITY__ 1
#define __FLT16_HAS_DENORM__ 1
#define __FLT16_MANT_DIG__ 11
#define __FLT16_DIG__ 3
#endif

// Default values.  Should be overridden in configuration files if necessary.
#ifndef __glibcxx_float16_has_denorm_loss
#  define __glibcxx_float16_has_denorm_loss false
#endif
#ifndef __glibcxx_float16_traps
#  define __glibcxx_float16_traps false
#endif
#ifndef __glibcxx_float16_tinyness_before
#  define __glibcxx_float16_tinyness_before false
#endif

namespace std {
  /// numeric_limits<xtd::xtd::float16_t> specialization.
  template<>
    struct numeric_limits<xtd::float16_t>
    {
      static constexpr bool is_specialized = true;
      static constexpr xtd::float16_t min() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x0400)); }
      static constexpr xtd::float16_t max() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7bff)); }
      static constexpr xtd::float16_t lowest() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0xfbff)); }

      static constexpr int digits = __FLT16_MANT_DIG__;
      static constexpr int digits10 = __FLT16_DIG__;
#define __glibcxx_max_digits10(T) (2 + (T) * 643L / 2136)
      static constexpr int max_digits10 = __glibcxx_max_digits10 (__FLT16_MANT_DIG__);
#undef __glibcxx_max_digits10
      static constexpr bool is_signed = true;
      static constexpr bool is_integer = false;
      static constexpr bool is_exact = false;
      static constexpr int radix = __FLT_RADIX__;

      static constexpr xtd::float16_t
      epsilon() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x1400)); }

      static constexpr xtd::float16_t round_error() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x3800)); }

      static constexpr int min_exponent = __FLT16_MIN_EXP__;
      static constexpr int min_exponent10 = __FLT16_MIN_10_EXP__;
      static constexpr int max_exponent = __FLT16_MAX_EXP__;
      static constexpr int max_exponent10 = __FLT16_MAX_10_EXP__;

      static constexpr bool has_infinity = __FLT16_HAS_INFINITY__;
      static constexpr bool has_quiet_NaN = __FLT16_HAS_QUIET_NAN__;
      static constexpr bool has_signaling_NaN = has_quiet_NaN;
      static constexpr float_denorm_style has_denorm = bool(__FLT16_HAS_DENORM__) ? denorm_present : denorm_absent;
      static constexpr bool has_denorm_loss = __glibcxx_float16_has_denorm_loss;

      static constexpr xtd::float16_t infinity() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7C00)); }
      static constexpr xtd::float16_t quiet_NaN() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7E00)); }
      static constexpr xtd::float16_t signaling_NaN() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x7D00)); }
      static constexpr xtd::float16_t denorm_min() noexcept { return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(0x0001)); }

      static constexpr bool is_iec559 = has_infinity && has_quiet_NaN && has_denorm == denorm_present;
      static constexpr bool is_bounded = true;
      static constexpr bool is_modulo = false;

      static constexpr bool traps = __glibcxx_float16_traps;
      static constexpr bool tinyness_before = __glibcxx_float16_tinyness_before;
      static constexpr float_round_style round_style = round_to_nearest;
    };

  XTD_DEVICE_FUNCTION inline constexpr int fpclassify(::xtd::float16_t arg) noexcept {
    uint16_t const bits = std::bit_cast<uint16_t>(arg);
    uint16_t const exp = bits & 0x7c00;
    uint16_t const mant = bits & 0x03ff;

    if (exp == 0) {
      return (mant == 0) ? FP_ZERO : FP_SUBNORMAL;
    }
    if (exp == 0x7c00) {
      return (mant == 0) ? FP_INFINITE : FP_NAN;
    }
    return FP_NORMAL;
  }

  XTD_DEVICE_FUNCTION inline constexpr bool isnan(::xtd::float16_t arg) noexcept {
    return (std::bit_cast<uint16_t>(arg) & 0x7c00) == 0x7c00 && (std::bit_cast<uint16_t>(arg) & 0x03ff) != 0;
  }

  XTD_DEVICE_FUNCTION inline constexpr bool isinf(::xtd::float16_t arg) noexcept {
    return (std::bit_cast<uint16_t>(arg) & 0x7fff) == 0x7c00;
  }

  XTD_DEVICE_FUNCTION inline constexpr xtd::float16_t abs(::xtd::float16_t arg) noexcept {
    return std::bit_cast<xtd::float16_t>(static_cast<uint16_t>(std::bit_cast<uint16_t>(arg) & 0x7fff));
  }
} // namespace std

#undef __glibcxx_float16_has_denorm_loss
#undef __glibcxx_float16_traps
#undef __glibcxx_float16_tinyness_before
