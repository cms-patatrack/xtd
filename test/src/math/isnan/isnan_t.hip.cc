/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/isnan.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "reference_isnan.h"

TEST_CASE("xtd::isnan", "[isnan][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isnan(float)") {
          validate<int, float, xtd::isnan, reference_isnan>(device);
        }

        SECTION("int xtd::isnan(double)") {
          validate<int, double, xtd::isnan, reference_isnan>(device);
        }

        SECTION("int xtd::isnan(int)") {
          validate<int, int, xtd::isnan, reference_isnan>(device);
        }
      }
    }
  }
}
