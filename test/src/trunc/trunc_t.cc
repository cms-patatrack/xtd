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
#include "xtd/math/trunc.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_trunc.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::trunc", "[trunc][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::trunc(float)") {
      validate<float, float, xtd::trunc, mpfr_truncf>(device, ulps_single);
    }

    SECTION("double xtd::trunc(double)") {
      validate<double, double, xtd::trunc, mpfr_trunc>(device, ulps_double);
    }

    SECTION("double xtd::trunc(int)") {
      validate<double, int, xtd::trunc, mpfr_trunc>(device, ulps_double);
    }

    SECTION("float xtd::truncf(float)") {
      validate<float, float, xtd::truncf, mpfr_truncf>(device, ulps_single);
    }

    SECTION("float xtd::truncf(double)") {
      validate<float, double, xtd::truncf, mpfr_truncf>(device, ulps_single);
    }

    SECTION("float xtd::truncf(int)") {
      validate<float, int, xtd::truncf, mpfr_truncf>(device, ulps_single);
    }
  }
}
