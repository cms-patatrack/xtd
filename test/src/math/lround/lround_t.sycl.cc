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
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_lround.h"

TEST_CASE("xtd::lround", "[lround][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("long xtd::lround(float)") {
            validate<long, float, xtd::lround, mpfr_lroundf>(platform, device);
          }

          SECTION("long xtd::lround(double)") {
            validate<long, double, xtd::lround, mpfr_lround>(platform, device);
          }

          SECTION("long xtd::lround(int)") {
            validate<long, int, xtd::lround, mpfr_lround>(platform, device);
          }

          SECTION("long xtd::lroundf(float)") {
            validate<long, float, xtd::lroundf, mpfr_lroundf>(platform, device);
          }

          SECTION("long xtd::lroundf(double)") {
            validate<long, double, xtd::lroundf, mpfr_lroundf>(platform, device);
          }

          SECTION("long xtd::lroundf(int)") {
            validate<long, int, xtd::lroundf, mpfr_lroundf>(platform, device);
          }
        }
      }
    }
  }
}
