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
#include "xtd/math/atan.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_atan.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::atan", "[atan][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::atan(float)") {
      validate<float, float, xtd::atan, mpfr_atanf>(device, ulps_single);
    }

    SECTION("double xtd::atan(double)") {
      validate<double, double, xtd::atan, mpfr_atan>(device, ulps_double);
    }

    SECTION("double xtd::atan(int)") {
      validate<double, int, xtd::atan, mpfr_atan>(device, ulps_double);
    }

    SECTION("float xtd::atanf(float)") {
      validate<float, float, xtd::atanf, mpfr_atanf>(device, ulps_single);
    }

    SECTION("float xtd::atanf(double)") {
      validate<float, double, xtd::atanf, mpfr_atanf>(device, ulps_single);
    }

    SECTION("float xtd::atanf(int)") {
      validate<float, int, xtd::atanf, mpfr_atanf>(device, ulps_single);
    }
  }
}
