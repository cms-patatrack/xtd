/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/div.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate_div.h"
#include "reference_div.h"

TEST_CASE("xtd::div", "[div][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("div_t xtd::div(int, int)") {
      validate_div<div_t, int, xtd::div, reference_div>(device);
    }
  }
}
