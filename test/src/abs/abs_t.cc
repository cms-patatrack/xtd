/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <cstdlib>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/abs.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate.h"

TEST_CASE("xtd::abs", "[abs][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("float xtd::abs(float)") {
      validate<float, float, xtd::abs, std::abs>(device);
    }

    SECTION("double xtd::abs(double)") {
      validate<double, double, xtd::abs, std::abs>(device);
    }

    SECTION("int xtd::abs(int)") {
      validate<int, int, xtd::abs, std::abs>(device);
    }

    SECTION("long xtd::abs(long)") {
      validate<long, long, xtd::abs, std::abs>(device);
    }

    SECTION("long long xtd::abs(long long)") {
      validate<long long, long long, xtd::abs, std::abs>(device);
    }
  }
}
