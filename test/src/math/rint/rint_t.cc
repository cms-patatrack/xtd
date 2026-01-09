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
#include "xtd/math/rint.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_rint.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::rint", "[rint][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::rint(float)") {
      validate<float, float, xtd::rint, mpfr_rintf>(device, ulps_single);
    }

    SECTION("double xtd::rint(double)") {
      validate<double, double, xtd::rint, mpfr_rint>(device, ulps_double);
    }

    SECTION("double xtd::rint(int)") {
      validate<double, int, xtd::rint, mpfr_rint>(device, ulps_double);
    }

    SECTION("float xtd::rintf(float)") {
      validate<float, float, xtd::rintf, mpfr_rintf>(device, ulps_single);
    }

    SECTION("float xtd::rintf(double)") {
      validate<float, double, xtd::rintf, mpfr_rintf>(device, ulps_single);
    }

    SECTION("float xtd::rintf(int)") {
      validate<float, int, xtd::rintf, mpfr_rintf>(device, ulps_single);
    }
  }
}
