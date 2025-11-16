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
#include "xtd/math/acosh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_acosh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::acosh", "[acosh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::acosh(float)") {
      validate<float, float, xtd::acosh, mpfr_acoshf>(device, ulps_single);
    }

    SECTION("double xtd::acosh(double)") {
      validate<double, double, xtd::acosh, mpfr_acosh>(device, ulps_double);
    }

    SECTION("double xtd::acosh(int)") {
      validate<double, int, xtd::acosh, mpfr_acosh>(device, ulps_double);
    }

    SECTION("float xtd::acoshf(float)") {
      validate<float, float, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
    }

    SECTION("float xtd::acoshf(double)") {
      validate<float, double, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
    }

    SECTION("float xtd::acoshf(int)") {
      validate<float, int, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
    }
  }
}
