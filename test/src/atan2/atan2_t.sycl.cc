/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atan2.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_atan2.h"

constexpr int ulps_single = 6;
constexpr int ulps_double = 6;

TEST_CASE("xtd::atan2", "[atan2][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::atan2(float, float)") {
            test::sycl::validate<float, float, xtd::atan2, mpfr_atan2f>(platform, device, ulps_single);
          }

          SECTION("double xtd::atan2(double, double)") {
            test::sycl::validate<double, double, xtd::atan2, mpfr_atan2>(platform, device, ulps_double);
          }

          SECTION("double xtd::atan2(int, int)") {
            test::sycl::validate<double, int, xtd::atan2, mpfr_atan2>(platform, device, ulps_double);
          }

          SECTION("float xtd::atan2f(float, float)") {
            test::sycl::validate<float, float, xtd::atan2f, mpfr_atan2f>(platform, device, ulps_single);
          }

          SECTION("float xtd::atan2f(double, double)") {
            test::sycl::validate<float, double, xtd::atan2f, mpfr_atan2f>(platform, device, ulps_single);
          }

          SECTION("float xtd::atan2f(int, int)") {
            test::sycl::validate<float, int, xtd::atan2f, mpfr_atan2f>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
