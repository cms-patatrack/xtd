/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <numbers>
#include <vector>

// MPFR headers
#include <mpfr.h>

// XTD headers
#include "xtd/internal/float16.h"
#include <xtd/concepts/arithmetic.h>

// Test helpers
#include "compare.h"

// std::nextafter is constexpr only in C++23 and newer
#if __cplusplus >= 202302L
#define constexpr_cxx23 constexpr
#else
#define constexpr_cxx23
#endif

// ---------------------------------------------------------------------------
// Correctly rounded binary16 references computed with MPFR
// ---------------------------------------------------------------------------

namespace test_half {

  // Precision of the intermediate MPFR result. The single rounding to binary16 below is correct
  // unless the exact result lies within about 2^-200 (relative) of a binary16 midpoint.
  inline constexpr mpfr_prec_t kRefPrec = 200;

  namespace detail {

    struct MpfrValue {
      mpfr_t value{};

      explicit MpfrValue(mpfr_prec_t prec) {
        mpfr_init2(static_cast<mpfr_ptr>(value), prec);
      }
      ~MpfrValue() {
        mpfr_clear(static_cast<mpfr_ptr>(value));
      }

      MpfrValue(const MpfrValue&) = delete;
      MpfrValue& operator=(const MpfrValue&) = delete;

      MpfrValue(MpfrValue&& other) noexcept {
        mpfr_init2(static_cast<mpfr_ptr>(value), 1);
        mpfr_swap(static_cast<mpfr_ptr>(value), static_cast<mpfr_ptr>(other.value));
      }

      MpfrValue& operator=(MpfrValue&& other) noexcept {
        if (this != &other) {
          mpfr_swap(static_cast<mpfr_ptr>(value), static_cast<mpfr_ptr>(other.value));
        }
        return *this;
      }

      // NOLINTBEGIN(cppcoreguidelines-explicit-constructor)
      operator mpfr_ptr() { return static_cast<mpfr_ptr>(value); }
      operator mpfr_srcptr() const { return static_cast<mpfr_srcptr>(value); }
      // NOLINTEND(cppcoreguidelines-explicit-constructor)
    };

  }  // namespace detail

  // Round v to the nearest binary16 value (ties to even, overflow to infinity)
  inline std::uint16_t round_to_half_bits(mpfr_srcptr v) {
    if (mpfr_nan_p(v)) {
      return 0x7e00;
    }
    const std::uint16_t sign = mpfr_signbit(v) ? 0x8000 : 0;
    if (mpfr_inf_p(v)) {
      // NOLINTNEXTLINE(bugprone-signed-bitwise)
      return sign | 0x7c00;
    }
    if (mpfr_zero_p(v)) {
      return sign;
    }

    // v = m * 2^e with 1/2 <= |m| < 1. In a normal binade binary16 has 11 significant bits, so
    // the unit in the last place is 2^(e-11), but never finer than the subnormal unit 2^-24
    const mpfr_exp_t e = mpfr_get_exp(v);
    mpfr_exp_t q = std::max<mpfr_exp_t>(e - 11, -24);

    // n = |v| / 2^q rounded to the nearest integer (ties to even)
    detail::MpfrValue w(kRefPrec);
    mpfr_abs(w, v, MPFR_RNDN);
    mpfr_mul_2si(w, w, -q, MPFR_RNDN);
    mpfr_rint(w, w, MPFR_RNDN);
    auto n = static_cast<unsigned long>(mpfr_get_ui(w, MPFR_RNDZ));

    if (q == -24) {
      // Subnormal range and the lowest normal binade. The binary16 encoding is linear in the
      // value here, so the bit pattern is n itself (n == 0x800 is 2^-13)
      return static_cast<std::uint16_t>(sign | n);
    }

    // Normal range. n is in [1024, 2048]. Rounding up to 2048 moves to the next binade
    if (n == 2048) {
      n = 1024;
      ++q;
    }
    const mpfr_exp_t unbiased = q + 10;
    if (unbiased > 15) {
      // NOLINTNEXTLINE(bugprone-signed-bitwise)
      return sign | 0x7c00;  // overflow to infinity
    }
    const auto biased = static_cast<unsigned long>(unbiased + 15);
    // NOLINTNEXTLINE(bugprone-signed-bitwise)
    return static_cast<std::uint16_t>(sign | (biased << 10) | (n - 1024));
  }

  // binary16 reference for an MPFR function
  template <typename Fn, xtd::arithmetic... Args>
    requires(sizeof...(Args) >= 1)
  xtd::float16_t reference(Fn fn, Args... args) {
    constexpr std::size_t kArity = sizeof...(Args);
    std::array<detail::MpfrValue, kArity> inputs = {[&args]() {
      detail::MpfrValue v(53);
      // Convert through binary16 first, so every input matches the half domain and is exact in double
      mpfr_set_d(v, static_cast<double>(static_cast<xtd::float16_t>(args)), MPFR_RNDN);
      return v;
    }()...};
    detail::MpfrValue out(kRefPrec);

    [&]<std::size_t... Is>(std::index_sequence<Is...>) {
      fn(out, static_cast<mpfr_srcptr>(inputs[Is])..., MPFR_RNDN);
    }(std::make_index_sequence<kArity>{});

    return std::bit_cast<xtd::float16_t>(round_to_half_bits(static_cast<mpfr_srcptr>(out)));
  }

}  // namespace test_half

// ---------------------------------------------------------------------------
// Input values
// ---------------------------------------------------------------------------

constexpr_cxx23 inline std::vector<double> generate_input_values() {
  std::vector<double> values;
  values.reserve(70000);

  // Every binary16 bit pattern: 65536 values, including +0, -0, subnormals, +inf, -inf and all NaN
  for (std::uint32_t bits = 0; bits <= 0xffff; ++bits) {
    values.push_back(static_cast<double>(std::bit_cast<xtd::float16_t>(static_cast<std::uint16_t>(bits))));
  }

  auto add_with_ulps = [&](double x) {
    values.push_back(x);

    // Double ULP shifts
    {
      double up = x;
      double down = x;
      for (int i = 0; i < 3; ++i) {
        up = std::nextafter(up, std::numeric_limits<double>::infinity());
        down = std::nextafter(down, -std::numeric_limits<double>::infinity());
        values.push_back(up);
        values.push_back(down);
      }
    }

    // Float ULP shifts (converted back to double)
    {
      float xf_up = static_cast<float>(x);
      float xf_down = xf_up;
      for (int i = 0; i < 3; ++i) {
        xf_up = std::nextafterf(xf_up, std::numeric_limits<float>::infinity());
        xf_down = std::nextafterf(xf_down, -std::numeric_limits<float>::infinity());
        values.push_back(static_cast<double>(xf_up));
        values.push_back(static_cast<double>(xf_down));
      }
    }
  };

  // Integers
  for (int i = -20; i <= 20; ++i) {
    add_with_ulps(static_cast<double>(i));
  }

  // Square roots
  for (int i = -20; i <= 20; ++i) {
    const double val = std::sqrt(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Multiples of pi/4
  for (int i = -25; i <= 25; ++i) {
    add_with_ulps(i * (std::numbers::pi / 4.0));
  }

  // Multiples of pi/3
  for (int i = -19; i <= 19; ++i) {
    add_with_ulps(i * (std::numbers::pi / 3.0));
  }

  // Logarithms
  for (int i = 1; i <= 20; ++i) {
    const double d = static_cast<double>(i);
    for (double val : {std::log(d), std::log2(d), std::log10(d)}) {
      add_with_ulps(val);
      add_with_ulps(-val);
    }
  }

  // Powers of e
  for (int i = -20; i <= 20; ++i) {
    const double val = std::exp(static_cast<double>(i));
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Powers of 2
  for (int i = -30; i <= 30; ++i) {
    const double val = std::ldexp(1.0, i);
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Powers of 10
  for (int i = -10; i <= 10; ++i) {
    const double val = std::pow(10.0, i);
    add_with_ulps(val);
    add_with_ulps(-val);
  }

  // Sort and remove duplicate entries, using bitwise ordering and comparisons to avoid problems with NaNs
  std::ranges::sort(values, [](double a, double b) { return std::bit_cast<std::uint64_t>(a) < std::bit_cast<std::uint64_t>(b); });
  auto last = std::ranges::unique(values, [](double a, double b) {
                return std::bit_cast<std::uint64_t>(a) == std::bit_cast<std::uint64_t>(b);
              }).begin();
  values.resize(static_cast<std::size_t>(std::distance(values.begin(), last)));

  return values;
}
