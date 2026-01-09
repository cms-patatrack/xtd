/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/cbrt.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_cbrt.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 1;

TEST_CASE("xtd::cbrt", "[cbrt][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::cbrt(float)") {
          validate<float, float, xtd::cbrt, mpfr_cbrtf>(device, ulps_single);
        }

        SECTION("double xtd::cbrt(double)") {
          validate<double, double, xtd::cbrt, mpfr_cbrt>(device, ulps_double);
        }

        SECTION("double xtd::cbrt(int)") {
          validate<double, int, xtd::cbrt, mpfr_cbrt>(device, ulps_double);
        }

        SECTION("float xtd::cbrtf(float)") {
          validate<float, float, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
        }

        SECTION("float xtd::cbrtf(double)") {
          validate<float, double, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
        }

        SECTION("float xtd::cbrtf(int)") {
          validate<float, int, xtd::cbrtf, mpfr_cbrtf>(device, ulps_single);
        }
      }
    }
  }
}
