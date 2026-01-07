/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/div.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate_div.h"
#include "reference_div.h"

TEST_CASE("xtd::div", "[fdim][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("div_t xtd::div(int, int)") {
          validate_div<div_t, int, xtd::div, reference_div>(device);
        }
      }
    }
  }
}
