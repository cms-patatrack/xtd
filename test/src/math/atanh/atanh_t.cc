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
#include "xtd/math/atanh.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_atanh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::atanh", "[atanh][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::atanh(float)") {
      validate<float, float, xtd::atanh, mpfr_atanhf>(device, ulps_single);
    }

    SECTION("double xtd::atanh(double)") {
      validate<double, double, xtd::atanh, mpfr_atanh>(device, ulps_double);
    }

    SECTION("double xtd::atanh(int)") {
      validate<double, int, xtd::atanh, mpfr_atanh>(device, ulps_double);
    }

    SECTION("float xtd::atanhf(float)") {
      validate<float, float, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
    }

    SECTION("float xtd::atanhf(double)") {
      validate<float, double, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
    }

    SECTION("float xtd::atanhf(int)") {
      validate<float, int, xtd::atanhf, mpfr_atanhf>(device, ulps_single);
    }
  }
}
