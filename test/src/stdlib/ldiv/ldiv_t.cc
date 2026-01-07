/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/ldiv.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate_div.h"
#include "reference_ldiv.h"

TEST_CASE("xtd::ldiv", "[ldiv][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("ldiv_t xtd::ldiv(long, long)") {
      validate_div<ldiv_t, long, xtd::ldiv, reference_ldiv>(device);
    }
  }
}
