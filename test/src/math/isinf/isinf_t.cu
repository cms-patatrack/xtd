/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/isinf.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "reference_isinf.h"

TEST_CASE("xtd::isinf", "[isinf][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::isinf(float)") {
          validate<int, float, xtd::isinf, reference_isinf>(device);
        }

        SECTION("int xtd::isinf(double)") {
          validate<int, double, xtd::isinf, reference_isinf>(device);
        }

        SECTION("int xtd::isinf(int)") {
          validate<int, int, xtd::isinf, reference_isinf>(device);
        }
      }
    }
  }
}
