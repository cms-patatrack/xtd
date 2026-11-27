/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <bit>
#include <concepts>
#include <cstdint>
#include <cstdlib>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// XTD headers
#include "xtd/internal/concepts.h"
#include "xtd/internal/float16.h"
#include "xtd/internal/limits.h"
#include "xtd/math/fabs.h"
#include "xtd/math/fpclassify.h"
#include "xtd/math/isinf.h"
#include "xtd/math/isnan.h"

template <xtd::floating_point T>
void compare(T result, T reference, int ulps = 0) {
  switch (xtd::fpclassify(reference)) {
    case FP_INFINITE:
      CHECK(xtd::isinf(result));
      break;
    case FP_NAN:
      CHECK(xtd::isnan(result));
      break;
    case FP_ZERO:
      CHECK_THAT(xtd::fabs(result), Catch::Matchers::WithinULP(static_cast<T>(0), ulps));
      break;
    case FP_SUBNORMAL:
      // Catch::Matchers::WithinULP does not handle properly the comparison of denormals with zero
      CHECK_THAT(result, Catch::Matchers::WithinAbs(reference, std::numeric_limits<T>::denorm_min() * ulps));
      break;
    case FP_NORMAL:
    default:
      CHECK_THAT(result, Catch::Matchers::WithinULP(reference, ulps));
  }
}

inline void compare(xtd::float16_t result, xtd::float16_t reference, int ulps = 0) {
  if (xtd::isnan(reference)) {
    CHECK(xtd::isnan(result));
    return;
  }
  if (xtd::isinf(reference)) {
    CHECK(xtd::isinf(result));
    return;
  }

  // NOLINTBEGIN(bugprone-signed-bitwise)
  // Map an IEEE 754 binary16 bit pattern to an integer that increases monotonically with the value.
  // -0 and +0 both map to 0. Finite values map to [-0x7bff, 0x7bff].
  constexpr auto to_ordered = [](uint16_t bits) {
    int32_t mag = bits & 0x7fff;
    return (bits & 0x8000) ? -mag : mag;
  };
  // NOLINTEND(bugprone-signed-bitwise)

  // Distance in half ULPs between two finite half values. Undefined for NaN and infinities.
  constexpr auto ulp_distance = [to_ordered](xtd::float16_t a, xtd::float16_t b) {
    return std::abs(to_ordered(std::bit_cast<uint16_t>(a)) - to_ordered(std::bit_cast<uint16_t>(b)));
  };

  // A NaN or infinite result must not pass against a finite reference
  CHECK(!xtd::isnan(result));
  CHECK(!xtd::isinf(result));
  CHECK(ulp_distance(result, reference) <= ulps);
}

template <std::integral T>
void compare(T result, T reference) {
  CHECK(result == reference);
}
