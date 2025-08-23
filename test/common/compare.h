/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

template <std::floating_point T>
void compare(T result, T reference) {
  switch (std::fpclassify(reference)) {
    case FP_INFINITE:
      CHECK(std::isinf(result));
      break;
    case FP_NAN:
      CHECK(std::isnan(result));
      break;
    case FP_ZERO:
      CHECK_THAT(result, Catch::Matchers::WithinULP(0., 1));
      break;
    case FP_NORMAL:
    case FP_SUBNORMAL:
    default:
      CHECK_THAT(result, Catch::Matchers::WithinULP(reference, 1));
  }
}
