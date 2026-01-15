/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/copysign.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "reference_copysign.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::copysign", "[copysign][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::copysign(float, float)") {
          validate<float, float, xtd::copysign, reference_copysignf>(device, ulps_single);
        }

        SECTION("double xtd::copysign(double, double)") {
          validate<double, double, xtd::copysign, reference_copysign>(device, ulps_double);
        }

        SECTION("double xtd::copysign(int, int)") {
          validate<double, int, xtd::copysign, reference_copysign>(device, ulps_double);
        }

        SECTION("float xtd::copysignf(float, float)") {
          validate<float, float, xtd::copysignf, reference_copysignf>(device, ulps_single);
        }

        SECTION("float xtd::copysignf(double, double)") {
          validate<float, double, xtd::copysignf, reference_copysignf>(device, ulps_single);
        }

        SECTION("float xtd::copysignf(int, int)") {
          validate<float, int, xtd::copysignf, reference_copysignf>(device, ulps_single);
        }
      }
    }
  }
}
