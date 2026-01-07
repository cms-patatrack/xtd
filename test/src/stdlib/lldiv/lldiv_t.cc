/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/lldiv.h"

// test headers
#include "common/cpu/device.h"
#include "common/cpu/validate_div.h"
#include "reference_lldiv.h"

TEST_CASE("xtd::lldiv", "[lldiv][cpu]") {
  const auto& device = test::cpu::device();
  DYNAMIC_SECTION("CPU: " << device.name()) {
    SECTION("lldiv_t xtd::lldiv(long long, long long)") {
      validate_div<lldiv_t, long long, xtd::lldiv, reference_lldiv>(device);
    }
  }
}
