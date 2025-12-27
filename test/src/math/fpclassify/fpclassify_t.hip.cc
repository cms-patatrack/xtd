/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fpclassify.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "reference_fpclassify.h"

TEST_CASE("xtd::fpclassify", "[fpclassify][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::fpclassify(float)") {
          validate<int, float, xtd::fpclassify, reference_fpclassify>(device);
        }

        SECTION("int xtd::fpclassify(double)") {
          validate<int, double, xtd::fpclassify, reference_fpclassify>(device);
        }

        SECTION("int xtd::fpclassify(int)") {
          validate<int, int, xtd::fpclassify, reference_fpclassify>(device);
        }
      }
    }
  }
}
