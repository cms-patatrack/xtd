/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/acosh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_acosh.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::acosh", "[acosh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::acosh(float)") {
          validate<float, float, xtd::acosh, mpfr_acoshf>(device, ulps_single);
        }

        SECTION("double xtd::acosh(double)") {
          validate<double, double, xtd::acosh, mpfr_acosh>(device, ulps_double);
        }

        SECTION("double xtd::acosh(int)") {
          validate<double, int, xtd::acosh, mpfr_acosh>(device, ulps_double);
        }

        SECTION("float xtd::acoshf(float)") {
          validate<float, float, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
        }

        SECTION("float xtd::acoshf(double)") {
          validate<float, double, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
        }

        SECTION("float xtd::acoshf(int)") {
          validate<float, int, xtd::acoshf, mpfr_acoshf>(device, ulps_single);
        }
      }
    }
  }
}
