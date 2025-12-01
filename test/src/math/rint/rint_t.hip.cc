/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/rint.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_rint.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::rint", "[rint][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::rint(float)") {
          validate<float, float, xtd::rint, mpfr_rintf>(device, ulps_single);
        }

        SECTION("double xtd::rint(double)") {
          validate<double, double, xtd::rint, mpfr_rint>(device, ulps_double);
        }

        SECTION("double xtd::rint(int)") {
          validate<double, int, xtd::rint, mpfr_rint>(device, ulps_double);
        }

        SECTION("float xtd::rintf(float)") {
          validate<float, float, xtd::rintf, mpfr_rintf>(device, ulps_single);
        }

        SECTION("float xtd::rintf(double)") {
          validate<float, double, xtd::rintf, mpfr_rintf>(device, ulps_single);
        }

        SECTION("float xtd::rintf(int)") {
          validate<float, int, xtd::rintf, mpfr_rintf>(device, ulps_single);
        }
      }
    }
  }
}
