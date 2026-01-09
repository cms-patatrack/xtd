/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/floor.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_floor.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::floor", "[floor][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::floor(float)") {
          validate<float, float, xtd::floor, mpfr_floorf>(device, ulps_single);
        }

        SECTION("double xtd::floor(double)") {
          validate<double, double, xtd::floor, mpfr_floor>(device, ulps_double);
        }

        SECTION("double xtd::floor(int)") {
          validate<double, int, xtd::floor, mpfr_floor>(device, ulps_double);
        }

        SECTION("float xtd::floorf(float)") {
          validate<float, float, xtd::floorf, mpfr_floorf>(device, ulps_single);
        }

        SECTION("float xtd::floorf(double)") {
          validate<float, double, xtd::floorf, mpfr_floorf>(device, ulps_single);
        }

        SECTION("float xtd::floorf(int)") {
          validate<float, int, xtd::floorf, mpfr_floorf>(device, ulps_single);
        }
      }
    }
  }
}
