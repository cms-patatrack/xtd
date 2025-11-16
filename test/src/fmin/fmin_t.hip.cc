/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/fmin.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_fmin.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::fmin", "[fmin][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::fmin(float, float)") {
          validate<float, float, xtd::fmin, mpfr_fminf>(device, ulps_single);
        }

        SECTION("double xtd::fmin(double, double)") {
          validate<double, double, xtd::fmin, mpfr_fmin>(device, ulps_double);
        }

        SECTION("double xtd::fmin(int, int)") {
          validate<double, int, xtd::fmin, mpfr_fmin>(device, ulps_double);
        }

        SECTION("float xtd::fminf(float, float)") {
          validate<float, float, xtd::fminf, mpfr_fminf>(device, ulps_single);
        }

        SECTION("float xtd::fminf(double, double)") {
          validate<float, double, xtd::fminf, mpfr_fminf>(device, ulps_single);
        }

        SECTION("float xtd::fminf(int, int)") {
          validate<float, int, xtd::fminf, mpfr_fminf>(device, ulps_single);
        }
      }
    }
  }
}
