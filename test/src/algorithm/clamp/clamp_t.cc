/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <algorithm>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/algorithm/clamp.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate_clamp.h"
#include "byval_clamp.h"

TEST_CASE("xtd::clamp", "[clamp][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::clamp(float, float)") {
      validate_clamp<float, float, xtd::clamp, byval::clamp>(device);
    }

    SECTION("double xtd::clamp(double, double)") {
      validate_clamp<double, double, xtd::clamp, byval::clamp>(device);
    }

    SECTION("int xtd::clamp(int, int)") {
      validate_clamp<int, int, xtd::clamp, byval::clamp>(device);
    }

    SECTION("long xtd::clamp(long, long)") {
      validate_clamp<long, long, xtd::clamp, byval::clamp>(device);
    }

    SECTION("long long xtd::clamp(long long, long long)") {
      validate_clamp<long long, long long, xtd::clamp, byval::clamp>(device);
    }
  }
}
