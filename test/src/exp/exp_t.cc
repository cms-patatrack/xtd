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
#include "xtd/math/exp.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_exp.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp", "[exp][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::exp(float)") {
      validate<float, float, xtd::exp, mpfr_expf>(device, ulps_single);
    }

    SECTION("double xtd::exp(double)") {
      validate<double, double, xtd::exp, mpfr_exp>(device, ulps_double);
    }

    SECTION("double xtd::exp(int)") {
      validate<double, int, xtd::exp, mpfr_exp>(device, ulps_double);
    }

    SECTION("float xtd::expf(float)") {
      validate<float, float, xtd::expf, mpfr_expf>(device, ulps_single);
    }

    SECTION("float xtd::expf(double)") {
      validate<float, double, xtd::expf, mpfr_expf>(device, ulps_single);
    }

    SECTION("float xtd::expf(int)") {
      validate<float, int, xtd::expf, mpfr_expf>(device, ulps_single);
    }
  }
}
