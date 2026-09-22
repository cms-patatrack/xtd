/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// XTD headers
#include "xtd/internal/concepts.h"
#include "xtd/internal/limits.h"
#include "xtd/math/fpclassify.h"
#include "xtd/math/isinf.h"
#include "xtd/math/isnan.h"
#include "xtd/math/fabs.h"

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

template <std::integral T>
void compare(T result, T reference) {
  CHECK(result == reference);
}
