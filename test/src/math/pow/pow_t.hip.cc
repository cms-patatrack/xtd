/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/pow.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_pow.h"

constexpr int ulps_single = 1;
constexpr int ulps_double = 1;

TEST_CASE("xtd::pow", "[pow][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::pow(float, float)") {
          validate<float, float, xtd::pow, mpfr_powf>(device, ulps_single);
        }

        SECTION("double xtd::pow(double, double)") {
          validate<double, double, xtd::pow, mpfr_pow>(device, ulps_double);
        }

        SECTION("double xtd::pow(int, int)") {
          validate<double, int, xtd::pow, mpfr_pow>(device, ulps_double);
        }

        SECTION("float xtd::powf(float, float)") {
          validate<float, float, xtd::powf, mpfr_powf>(device, ulps_single);
        }

        SECTION("float xtd::powf(double, double)") {
          validate<float, double, xtd::powf, mpfr_powf>(device, ulps_single);
        }

        SECTION("float xtd::powf(int, int)") {
          validate<float, int, xtd::powf, mpfr_powf>(device, ulps_single);
        }
      }
    }
  }
}
