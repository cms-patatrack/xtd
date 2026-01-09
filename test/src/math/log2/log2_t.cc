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
#include "xtd/math/log2.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_log2.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 2;

TEST_CASE("xtd::log2", "[log2][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::log2(float)") {
      validate<float, float, xtd::log2, mpfr_log2f>(device, ulps_single);
    }

    SECTION("double xtd::log2(double)") {
      validate<double, double, xtd::log2, mpfr_log2>(device, ulps_double);
    }

    SECTION("double xtd::log2(int)") {
      validate<double, int, xtd::log2, mpfr_log2>(device, ulps_double);
    }

    SECTION("float xtd::log2f(float)") {
      validate<float, float, xtd::log2f, mpfr_log2f>(device, ulps_single);
    }

    SECTION("float xtd::log2f(double)") {
      validate<float, double, xtd::log2f, mpfr_log2f>(device, ulps_single);
    }

    SECTION("float xtd::log2f(int)") {
      validate<float, int, xtd::log2f, mpfr_log2f>(device, ulps_single);
    }
  }
}
