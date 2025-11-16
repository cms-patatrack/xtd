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
#include "xtd/math/cos.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_cos.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::cos", "[cos][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::cos(float)") {
      validate<float, float, xtd::cos, mpfr_cosf>(device, ulps_single);
    }

    SECTION("double xtd::cos(double)") {
      validate<double, double, xtd::cos, mpfr_cos>(device, ulps_double);
    }

    SECTION("double xtd::cos(int)") {
      validate<double, int, xtd::cos, mpfr_cos>(device, ulps_double);
    }

    SECTION("float xtd::cosf(float)") {
      validate<float, float, xtd::cosf, mpfr_cosf>(device, ulps_single);
    }

    SECTION("float xtd::cosf(double)") {
      validate<float, double, xtd::cosf, mpfr_cosf>(device, ulps_single);
    }

    SECTION("float xtd::cosf(int)") {
      validate<float, int, xtd::cosf, mpfr_cosf>(device, ulps_single);
    }
  }
}
