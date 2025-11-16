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
#include "xtd/math/exp2.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_exp2.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp2", "[exp2][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::exp2(float)") {
      validate<float, float, xtd::exp2, mpfr_exp2f>(device, ulps_single);
    }

    SECTION("double xtd::exp2(double)") {
      validate<double, double, xtd::exp2, mpfr_exp2>(device, ulps_double);
    }

    SECTION("double xtd::exp2(int)") {
      validate<double, int, xtd::exp2, mpfr_exp2>(device, ulps_double);
    }

    SECTION("float xtd::exp2f(float)") {
      validate<float, float, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
    }

    SECTION("float xtd::exp2f(double)") {
      validate<float, double, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
    }

    SECTION("float xtd::exp2f(int)") {
      validate<float, int, xtd::exp2f, mpfr_exp2f>(device, ulps_single);
    }
  }
}
