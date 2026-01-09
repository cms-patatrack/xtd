/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/remquo.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate_remquo.h"
#include "mpfr_remquo.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::remquo", "[remquo][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::remquo(float, float, int*)") {
            test::sycl::validate_remquo<float, float, xtd::remquo, mpfr_remquof>(platform, device, ulps_single);
          }

          SECTION("double xtd::remquo(double, double, int*)") {
            test::sycl::validate_remquo<double, double, xtd::remquo, mpfr_remquo>(platform, device, ulps_double);
          }

          SECTION("double xtd::remquo(int, int, int*)") {
            test::sycl::validate_remquo<double, int, xtd::remquo, mpfr_remquo>(platform, device, ulps_double);
          }

          SECTION("float xtd::remquof(float, float, int*)") {
            test::sycl::validate_remquo<float, float, xtd::remquof, mpfr_remquof>(platform, device, ulps_single);
          }

          SECTION("float xtd::remquof(double, double, int*)") {
            test::sycl::validate_remquo<float, double, xtd::remquof, mpfr_remquof>(platform, device, ulps_single);
          }

          SECTION("float xtd::remquof(int, int, int*)") {
            test::sycl::validate_remquo<float, int, xtd::remquof, mpfr_remquof>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
