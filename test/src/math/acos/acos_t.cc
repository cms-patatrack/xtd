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
#include "xtd/math/acos.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_acos.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::acos", "[acos][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::acos(float)") {
      validate<float, float, xtd::acos, mpfr_acosf>(device, ulps_single);
    }

    SECTION("double xtd::acos(double)") {
      validate<double, double, xtd::acos, mpfr_acos>(device, ulps_double);
    }

    SECTION("double xtd::acos(int)") {
      validate<double, int, xtd::acos, mpfr_acos>(device, ulps_double);
    }

    SECTION("float xtd::acosf(float)") {
      validate<float, float, xtd::acosf, mpfr_acosf>(device, ulps_single);
    }

    SECTION("float xtd::acosf(double)") {
      validate<float, double, xtd::acosf, mpfr_acosf>(device, ulps_single);
    }

    SECTION("float xtd::acosf(int)") {
      validate<float, int, xtd::acosf, mpfr_acosf>(device, ulps_single);
    }
  }
}
