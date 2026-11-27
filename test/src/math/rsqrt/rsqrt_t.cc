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
#include "xtd/math/rsqrt.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_rsqrt.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::rsqrt", "[rsqrt][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::rsqrt(float)") {
      validate<float, float, xtd::rsqrt, mpfr_rsqrtf>(device, ulps_single);
    }

    SECTION("double xtd::rsqrt(double)") {
      validate<double, double, xtd::rsqrt, mpfr_rsqrt>(device, ulps_double);
    }

    SECTION("double xtd::rsqrt(int)") {
      validate<double, int, xtd::rsqrt, mpfr_rsqrt>(device, ulps_double);
    }

    SECTION("float xtd::rsqrtf(float)") {
      validate<float, float, xtd::rsqrtf, mpfr_rsqrtf>(device, ulps_single);
    }

    SECTION("float xtd::rsqrtf(double)") {
      validate<float, double, xtd::rsqrtf, mpfr_rsqrtf>(device, ulps_single);
    }

    SECTION("float xtd::rsqrtf(int)") {
      validate<float, int, xtd::rsqrtf, mpfr_rsqrtf>(device, ulps_single);
    }
  }
}
