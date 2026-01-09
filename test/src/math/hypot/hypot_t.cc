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
#include "xtd/math/hypot.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_hypot.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::hypot", "[hypot][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::hypot(float, float)") {
      validate<float, float, xtd::hypot, mpfr_hypotf>(device, ulps_single);
    }

    SECTION("double xtd::hypot(double, double)") {
      validate<double, double, xtd::hypot, mpfr_hypot>(device, ulps_double);
    }

    SECTION("double xtd::hypot(int, int)") {
      validate<double, int, xtd::hypot, mpfr_hypot>(device, ulps_double);
    }

    SECTION("float xtd::hypotf(float, float)") {
      validate<float, float, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
    }

    SECTION("float xtd::hypotf(double, double)") {
      validate<float, double, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
    }

    SECTION("float xtd::hypotf(int, int)") {
      validate<float, int, xtd::hypotf, mpfr_hypotf>(device, ulps_single);
    }
  }
}
