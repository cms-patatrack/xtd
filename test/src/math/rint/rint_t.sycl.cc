/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/rint.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_rint.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::rint", "[rint][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::rint(float)") {
            validate<float, float, xtd::rint, mpfr_rintf>(platform, device, ulps_single);
          }

          SECTION("double xtd::rint(double)") {
            validate<double, double, xtd::rint, mpfr_rint>(platform, device, ulps_double);
          }

          SECTION("double xtd::rint(int)") {
            validate<double, int, xtd::rint, mpfr_rint>(platform, device, ulps_double);
          }

          SECTION("float xtd::rintf(float)") {
            validate<float, float, xtd::rintf, mpfr_rintf>(platform, device, ulps_single);
          }

          SECTION("float xtd::rintf(double)") {
            validate<float, double, xtd::rintf, mpfr_rintf>(platform, device, ulps_single);
          }

          SECTION("float xtd::rintf(int)") {
            validate<float, int, xtd::rintf, mpfr_rintf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
