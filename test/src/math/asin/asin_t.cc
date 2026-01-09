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
#include "xtd/math/asin.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_asin.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::asin", "[asin][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::asin(float)") {
      validate<float, float, xtd::asin, mpfr_asinf>(device, ulps_single);
    }

    SECTION("double xtd::asin(double)") {
      validate<double, double, xtd::asin, mpfr_asin>(device, ulps_double);
    }

    SECTION("double xtd::asin(int)") {
      validate<double, int, xtd::asin, mpfr_asin>(device, ulps_double);
    }

    SECTION("float xtd::asinf(float)") {
      validate<float, float, xtd::asinf, mpfr_asinf>(device, ulps_single);
    }

    SECTION("float xtd::asinf(double)") {
      validate<float, double, xtd::asinf, mpfr_asinf>(device, ulps_single);
    }

    SECTION("float xtd::asinf(int)") {
      validate<float, int, xtd::asinf, mpfr_asinf>(device, ulps_single);
    }
  }
}
