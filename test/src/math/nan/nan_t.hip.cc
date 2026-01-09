/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/nan.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "reference_nan.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::nan", "[nan][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::nanf(uint32_t)") {
          validate<float, uint32_t, xtd::nanf, reference_nanf>(device);
        }

        SECTION("double xtd::nan(uint64_t)") {
          validate<double, uint64_t, xtd::nan, reference_nan>(device);
        }
      }
    }
  }
}
