/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/cosh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_cosh.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::cosh", "[cosh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::cosh(float)") {
          validate<float, float, xtd::cosh, mpfr_coshf>(device, ulps_single);
        }

        SECTION("double xtd::cosh(double)") {
          validate<double, double, xtd::cosh, mpfr_cosh>(device, ulps_double);
        }

        SECTION("double xtd::cosh(int)") {
          validate<double, int, xtd::cosh, mpfr_cosh>(device, ulps_double);
        }

        SECTION("float xtd::coshf(float)") {
          validate<float, float, xtd::coshf, mpfr_coshf>(device, ulps_single);
        }

        SECTION("float xtd::coshf(double)") {
          validate<float, double, xtd::coshf, mpfr_coshf>(device, ulps_single);
        }

        SECTION("float xtd::coshf(int)") {
          validate<float, int, xtd::coshf, mpfr_coshf>(device, ulps_single);
        }
      }
    }
  }
}
