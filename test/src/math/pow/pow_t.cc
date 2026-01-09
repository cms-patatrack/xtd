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
#include "xtd/math/pow.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_pow.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::pow", "[pow][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::pow(float, float)") {
      validate<float, float, xtd::pow, mpfr_powf>(device, ulps_single);
    }

    SECTION("double xtd::pow(double, double)") {
      validate<double, double, xtd::pow, mpfr_pow>(device, ulps_double);
    }

    SECTION("double xtd::pow(int, int)") {
      validate<double, int, xtd::pow, mpfr_pow>(device, ulps_double);
    }

    SECTION("float xtd::powf(float, float)") {
      validate<float, float, xtd::powf, mpfr_powf>(device, ulps_single);
    }

    SECTION("float xtd::powf(double, double)") {
      validate<float, double, xtd::powf, mpfr_powf>(device, ulps_single);
    }

    SECTION("float xtd::powf(int, int)") {
      validate<float, int, xtd::powf, mpfr_powf>(device, ulps_single);
    }
  }
}
