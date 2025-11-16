/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cmath>
#include <vector>

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/math/floor.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_floor.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::floor", "[floor][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::floor(float)") {
      validate<float, float, xtd::floor, mpfr_floorf>(device, ulps_single);
    }

    SECTION("double xtd::floor(double)") {
      validate<double, double, xtd::floor, mpfr_floor>(device, ulps_double);
    }

    SECTION("double xtd::floor(int)") {
      validate<double, int, xtd::floor, mpfr_floor>(device, ulps_double);
    }

    SECTION("float xtd::floorf(float)") {
      validate<float, float, xtd::floorf, mpfr_floorf>(device, ulps_single);
    }

    SECTION("float xtd::floorf(double)") {
      validate<float, double, xtd::floorf, mpfr_floorf>(device, ulps_single);
    }

    SECTION("float xtd::floorf(int)") {
      validate<float, int, xtd::floorf, mpfr_floorf>(device, ulps_single);
    }
  }
}
