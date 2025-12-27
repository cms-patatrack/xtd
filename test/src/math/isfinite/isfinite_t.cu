/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/isfinite.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "reference_isfinite.h"

TEST_CASE("xtd::isfinite", "[isfinite][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isfinite(float)") {
          validate<int, float, xtd::isfinite, reference_isfinite>(device);
        }

        SECTION("int xtd::isfinite(double)") {
          validate<int, double, xtd::isfinite, reference_isfinite>(device);
        }

        SECTION("int xtd::isfinite(int)") {
          validate<int, int, xtd::isfinite, reference_isfinite>(device);
        }
      }
    }
  }
}
