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
#include "xtd/math/ceil.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_ceil.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::ceil", "[ceil][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::ceil(float)") {
      validate<float, float, xtd::ceil, mpfr_ceilf>(device, ulps_single);
    }

    SECTION("double xtd::ceil(double)") {
      validate<double, double, xtd::ceil, mpfr_ceil>(device, ulps_double);
    }

    SECTION("double xtd::ceil(int)") {
      validate<double, int, xtd::ceil, mpfr_ceil>(device, ulps_double);
    }

    SECTION("float xtd::ceilf(float)") {
      validate<float, float, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
    }

    SECTION("float xtd::ceilf(double)") {
      validate<float, double, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
    }

    SECTION("float xtd::ceilf(int)") {
      validate<float, int, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
    }
  }
}
