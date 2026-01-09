/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/sin.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_sin.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::sin", "[sin][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::sin(float)") {
          validate<float, float, xtd::sin, mpfr_sinf>(device, ulps_single);
        }

        SECTION("double xtd::sin(double)") {
          validate<double, double, xtd::sin, mpfr_sin>(device, ulps_double);
        }

        SECTION("double xtd::sin(int)") {
          validate<double, int, xtd::sin, mpfr_sin>(device, ulps_double);
        }

        SECTION("float xtd::sinf(float)") {
          validate<float, float, xtd::sinf, mpfr_sinf>(device, ulps_single);
        }

        SECTION("float xtd::sinf(double)") {
          validate<float, double, xtd::sinf, mpfr_sinf>(device, ulps_single);
        }

        SECTION("float xtd::sinf(int)") {
          validate<float, int, xtd::sinf, mpfr_sinf>(device, ulps_single);
        }
      }
    }
  }
}
