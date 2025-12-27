/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/signbit.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "reference_signbit.h"

TEST_CASE("xtd::signbit", "[signbit][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("int xtd::signbit(float)") {
          validate<int, float, xtd::signbit, reference_signbit>(device);
        }

        SECTION("int xtd::signbit(double)") {
          validate<int, double, xtd::signbit, reference_signbit>(device);
        }

        SECTION("int xtd::signbit(int)") {
          validate<int, int, xtd::signbit, reference_signbit>(device);
        }

        SECTION("int xtd::signbit(unsigned int)") {
          validate<int, unsigned int, xtd::signbit, reference_signbit>(device);
        }
      }
    }
  }
}
