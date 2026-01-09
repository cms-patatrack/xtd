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
#include "xtd/math/remainder.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_remainder.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::remainder", "[remainder][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::remainder(float, float)") {
      validate<float, float, xtd::remainder, mpfr_remainderf>(device, ulps_single);
    }

    SECTION("double xtd::remainder(double, double)") {
      validate<double, double, xtd::remainder, mpfr_remainder>(device, ulps_double);
    }

    SECTION("double xtd::remainder(int, int)") {
      validate<double, int, xtd::remainder, mpfr_remainder>(device, ulps_double);
    }

    SECTION("float xtd::remainderf(float, float)") {
      validate<float, float, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
    }

    SECTION("float xtd::remainderf(double, double)") {
      validate<float, double, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
    }

    SECTION("float xtd::remainderf(int, int)") {
      validate<float, int, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
    }
  }
}
