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
#include "xtd/math/log10.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_log10.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log10", "[log10][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::log10(float)") {
      validate<float, float, xtd::log10, mpfr_log10f>(device, ulps_single);
    }

    SECTION("double xtd::log10(double)") {
      validate<double, double, xtd::log10, mpfr_log10>(device, ulps_double);
    }

    SECTION("double xtd::log10(int)") {
      validate<double, int, xtd::log10, mpfr_log10>(device, ulps_double);
    }

    SECTION("float xtd::log10f(float)") {
      validate<float, float, xtd::log10f, mpfr_log10f>(device, ulps_single);
    }

    SECTION("float xtd::log10f(double)") {
      validate<float, double, xtd::log10f, mpfr_log10f>(device, ulps_single);
    }

    SECTION("float xtd::log10f(int)") {
      validate<float, int, xtd::log10f, mpfr_log10f>(device, ulps_single);
    }
  }
}
