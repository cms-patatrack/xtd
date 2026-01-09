/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/asinh.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_asinh.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::asinh", "[asinh][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::asinh(float)") {
          validate<float, float, xtd::asinh, mpfr_asinhf>(device, ulps_single);
        }

        SECTION("double xtd::asinh(double)") {
          validate<double, double, xtd::asinh, mpfr_asinh>(device, ulps_double);
        }

        SECTION("double xtd::asinh(int)") {
          validate<double, int, xtd::asinh, mpfr_asinh>(device, ulps_double);
        }

        SECTION("float xtd::asinhf(float)") {
          validate<float, float, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
        }

        SECTION("float xtd::asinhf(double)") {
          validate<float, double, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
        }

        SECTION("float xtd::asinhf(int)") {
          validate<float, int, xtd::asinhf, mpfr_asinhf>(device, ulps_single);
        }
      }
    }
  }
}
