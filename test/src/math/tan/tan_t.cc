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
#include "xtd/math/tan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_tan.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::tan", "[tan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::tan(float)") {
      validate<float, float, xtd::tan, mpfr_tanf>(device, ulps_single);
    }

    SECTION("double xtd::tan(double)") {
      validate<double, double, xtd::tan, mpfr_tan>(device, ulps_double);
    }

    SECTION("double xtd::tan(int)") {
      validate<double, int, xtd::tan, mpfr_tan>(device, ulps_double);
    }

    SECTION("float xtd::tanf(float)") {
      validate<float, float, xtd::tanf, mpfr_tanf>(device, ulps_single);
    }

    SECTION("float xtd::tanf(double)") {
      validate<float, double, xtd::tanf, mpfr_tanf>(device, ulps_single);
    }

    SECTION("float xtd::tanf(int)") {
      validate<float, int, xtd::tanf, mpfr_tanf>(device, ulps_single);
    }
  }
}
