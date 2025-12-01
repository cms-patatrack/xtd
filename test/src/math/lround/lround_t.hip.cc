/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/lround.h"

// test headers
#include "common/hip/platform.h"
#include "common/hip/validate.h"
#include "mpfr_lround.h"

TEST_CASE("xtd::lround", "[lround][hip]") {
  const auto& platform = test::hip::platform();
  DYNAMIC_SECTION("HIP platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("HIP device " << device.index() << ": " << device.name()) {
        SECTION("long xtd::lround(float)") {
          validate<long, float, xtd::lround, mpfr_lroundf>(device);
        }

        SECTION("long xtd::lround(double)") {
          validate<long, double, xtd::lround, mpfr_lround>(device);
        }

        SECTION("long xtd::lround(int)") {
          validate<long, int, xtd::lround, mpfr_lround>(device);
        }

        SECTION("long xtd::lroundf(float)") {
          validate<long, float, xtd::lroundf, mpfr_lroundf>(device);
        }

        SECTION("long xtd::lroundf(double)") {
          validate<long, double, xtd::lroundf, mpfr_lroundf>(device);
        }

        SECTION("long xtd::lroundf(int)") {
          validate<long, int, xtd::lroundf, mpfr_lroundf>(device);
        }
      }
    }
  }
}
