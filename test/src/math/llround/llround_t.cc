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
#include "xtd/math/llround.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_llround.h"

TEST_CASE("xtd::llround", "[llround][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("long long xtd::llround(float)") {
      validate<long long, float, xtd::llround, mpfr_llroundf>(device);
    }

    SECTION("long long xtd::llround(double)") {
      validate<long long, double, xtd::llround, mpfr_llround>(device);
    }

    SECTION("long long xtd::llround(int)") {
      validate<long long, int, xtd::llround, mpfr_llround>(device);
    }

    SECTION("long long xtd::llroundf(float)") {
      validate<long long, float, xtd::llroundf, mpfr_llroundf>(device);
    }

    SECTION("long long xtd::llroundf(double)") {
      validate<long long, double, xtd::llroundf, mpfr_llroundf>(device);
    }

    SECTION("long long xtd::llroundf(int)") {
      validate<long long, int, xtd::llroundf, mpfr_llroundf>(device);
    }
  }
}
