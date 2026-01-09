/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/remainder.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_remainder.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::remainder", "[remainder][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::remainder(float, float)") {
          validate<float, float, xtd::remainder, mpfr_remainderf>(device, ulps_single);
        }

        SECTION("double xtd::remainder(double, double)") {
          validate<double, double, xtd::remainder, mpfr_remainder>(device, ulps_double);
        }

        SECTION("double xtd::remainder(int, int)") {
          validate<double, int, xtd::remainder, mpfr_remainder>(device, ulps_double);
        }

        SECTION("float xtd::remainderf(float, float)") {
          validate<float, float, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
        }

        SECTION("float xtd::remainderf(double, double)") {
          validate<float, double, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
        }

        SECTION("float xtd::remainderf(int, int)") {
          validate<float, int, xtd::remainderf, mpfr_remainderf>(device, ulps_single);
        }
      }
    }
  }
}
