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
#include "xtd/math/llrint.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"
#include "mpfr_llrint.h"

TEST_CASE("xtd::llrint", "[llrint][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("long long xtd::llrint(float)") {
      validate<long long, float, xtd::llrint, mpfr_llrintf>(device);
    }

    SECTION("long long xtd::llrint(double)") {
      validate<long long, double, xtd::llrint, mpfr_llrint>(device);
    }

    SECTION("long long xtd::llrint(int)") {
      validate<long long, int, xtd::llrint, mpfr_llrint>(device);
    }

    SECTION("long long xtd::llrintf(float)") {
      validate<long long, float, xtd::llrintf, mpfr_llrintf>(device);
    }

    SECTION("long long xtd::llrintf(double)") {
      validate<long long, double, xtd::llrintf, mpfr_llrintf>(device);
    }

    SECTION("long long xtd::llrintf(int)") {
      validate<long long, int, xtd::llrintf, mpfr_llrintf>(device);
    }
  }
}
