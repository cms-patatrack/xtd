/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/remainder.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_remainder.h"

constexpr int ulps_single = 2;  // 0 ULP according to the documentation
constexpr int ulps_double = 0;

TEST_CASE("xtd::remainder", "[remainder][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::remainder(float, float)") {
            test::sycl::validate<float, float, xtd::remainder, mpfr_remainderf>(platform, device, ulps_single);
          }

          SECTION("double xtd::remainder(double, double)") {
            test::sycl::validate<double, double, xtd::remainder, mpfr_remainder>(platform, device, ulps_double);
          }

          SECTION("double xtd::remainder(int, int)") {
            test::sycl::validate<double, int, xtd::remainder, mpfr_remainder>(platform, device, ulps_double);
          }

          SECTION("float xtd::remainderf(float, float)") {
            test::sycl::validate<float, float, xtd::remainderf, mpfr_remainderf>(platform, device, ulps_single);
          }

          SECTION("float xtd::remainderf(double, double)") {
            test::sycl::validate<float, double, xtd::remainderf, mpfr_remainderf>(platform, device, ulps_single);
          }

          SECTION("float xtd::remainderf(int, int)") {
            test::sycl::validate<float, int, xtd::remainderf, mpfr_remainderf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
