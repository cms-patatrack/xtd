/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/stdlib/ldiv.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate_div.h"
#include "reference_ldiv.h"

TEST_CASE("xtd::ldiv", "[fdim][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("ldiv_t xtd::ldiv(long, long)") {
          validate_div<ldiv_t, long, xtd::ldiv, reference_ldiv>(device);
        }
      }
    }
  }
}
