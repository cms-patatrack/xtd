/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/isnormal.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "reference_isnormal.h"

TEST_CASE("xtd::isnormal", "[isnormal][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isnormal(float)") {
          validate<int, float, xtd::isnormal, reference_isnormal>(device);
        }

        SECTION("int xtd::isnormal(double)") {
          validate<int, double, xtd::isnormal, reference_isnormal>(device);
        }

        SECTION("int xtd::isnormal(int)") {
          validate<int, int, xtd::isnormal, reference_isnormal>(device);
        }
      }
    }
  }
}
