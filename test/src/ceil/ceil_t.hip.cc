/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/ceil.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_ceil.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::ceil", "[ceil][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::ceil(float)") {
          validate<float, float, xtd::ceil, mpfr_ceilf>(device, ulps_single);
        }

        SECTION("double xtd::ceil(double)") {
          validate<double, double, xtd::ceil, mpfr_ceil>(device, ulps_double);
        }

        SECTION("double xtd::ceil(int)") {
          validate<double, int, xtd::ceil, mpfr_ceil>(device, ulps_double);
        }

        SECTION("float xtd::ceilf(float)") {
          validate<float, float, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
        }

        SECTION("float xtd::ceilf(double)") {
          validate<float, double, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
        }

        SECTION("float xtd::ceilf(int)") {
          validate<float, int, xtd::ceilf, mpfr_ceilf>(device, ulps_single);
        }
      }
    }
  }
}
