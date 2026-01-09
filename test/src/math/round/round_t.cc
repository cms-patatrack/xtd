/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/round.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_round.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::round", "[round][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::round(float)") {
      validate<float, float, xtd::round, mpfr_roundf>(device, ulps_single);
    }

    SECTION("double xtd::round(double)") {
      validate<double, double, xtd::round, mpfr_round>(device, ulps_double);
    }

    SECTION("double xtd::round(int)") {
      validate<double, int, xtd::round, mpfr_round>(device, ulps_double);
    }

    SECTION("float xtd::roundf(float)") {
      validate<float, float, xtd::roundf, mpfr_roundf>(device, ulps_single);
    }

    SECTION("float xtd::roundf(double)") {
      validate<float, double, xtd::roundf, mpfr_roundf>(device, ulps_single);
    }

    SECTION("float xtd::roundf(int)") {
      validate<float, int, xtd::roundf, mpfr_roundf>(device, ulps_single);
    }
  }
}
