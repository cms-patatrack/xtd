/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/expm1.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_expm1.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 2;  // 1 ULP for x ∈ [-10., +10.] according to the documentation

TEST_CASE("xtd::expm1", "[expm1][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::expm1(float)") {
          validate<float, float, xtd::expm1, mpfr_expm1f>(device, ulps_single);
        }

        SECTION("double xtd::expm1(double)") {
          validate<double, double, xtd::expm1, mpfr_expm1>(device, ulps_double);
        }

        SECTION("double xtd::expm1(int)") {
          validate<double, int, xtd::expm1, mpfr_expm1>(device, ulps_double);
        }

        SECTION("float xtd::expm1f(float)") {
          validate<float, float, xtd::expm1f, mpfr_expm1f>(device, ulps_single);
        }

        SECTION("float xtd::expm1f(double)") {
          validate<float, double, xtd::expm1f, mpfr_expm1f>(device, ulps_single);
        }

        SECTION("float xtd::expm1f(int)") {
          validate<float, int, xtd::expm1f, mpfr_expm1f>(device, ulps_single);
        }
      }
    }
  }
}
