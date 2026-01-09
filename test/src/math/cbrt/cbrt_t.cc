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
#include "xtd/math/cbrt.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_cbrt.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 4;

TEST_CASE("xtd::cbrt", "[cbrt][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::cbrt(float)") {
      validate<float, float, xtd::cbrt, mpfr_cbrtf>(device, ulps_single);
    }

    SECTION("double xtd::cbrt(double)") {
      validate<double, double, xtd::cbrt, mpfr_cbrt>(device, ulps_double);
    }

    SECTION("double xtd::cbrt(int)") {
      validate<double, int, xtd::cbrt, mpfr_cbrt>(device, ulps_double);
    }

    SECTION("float xtd::cbrtf(float)") {
      validate<float, float, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
    }

    SECTION("float xtd::cbrtf(double)") {
      validate<float, double, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
    }

    SECTION("float xtd::cbrtf(int)") {
      validate<float, int, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
    }
  }
}
