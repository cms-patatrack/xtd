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
#include "xtd/math/cosh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_cosh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::cosh", "[cosh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::cosh(float)") {
      validate<float, float, xtd::cosh, mpfr_coshf>(device, ulps_single);
    }

    SECTION("double xtd::cosh(double)") {
      validate<double, double, xtd::cosh, mpfr_cosh>(device, ulps_double);
    }

    SECTION("double xtd::cosh(int)") {
      validate<double, int, xtd::cosh, mpfr_cosh>(device, ulps_double);
    }

    SECTION("float xtd::coshf(float)") {
      validate<float, float, xtd::coshf, mpfr_coshf>(device, ulps_single);
    }

    SECTION("float xtd::coshf(double)") {
      validate<float, double, xtd::coshf, mpfr_coshf>(device, ulps_single);
    }

    SECTION("float xtd::coshf(int)") {
      validate<float, int, xtd::coshf, mpfr_coshf>(device, ulps_single);
    }
  }
}
