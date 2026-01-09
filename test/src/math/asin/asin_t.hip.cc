/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/asin.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_asin.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 1;

TEST_CASE("xtd::asin", "[asin][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::asin(float)") {
          validate<float, float, xtd::asin, mpfr_asinf>(device, ulps_single);
        }

        SECTION("double xtd::asin(double)") {
          validate<double, double, xtd::asin, mpfr_asin>(device, ulps_double);
        }

        SECTION("double xtd::asin(int)") {
          validate<double, int, xtd::asin, mpfr_asin>(device, ulps_double);
        }

        SECTION("float xtd::asinf(float)") {
          validate<float, float, xtd::asinf, mpfr_asinf>(device, ulps_single);
        }

        SECTION("float xtd::asinf(double)") {
          validate<float, double, xtd::asinf, mpfr_asinf>(device, ulps_single);
        }

        SECTION("float xtd::asinf(int)") {
          validate<float, int, xtd::asinf, mpfr_asinf>(device, ulps_single);
        }
      }
    }
  }
}
