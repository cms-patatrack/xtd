/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/sqrt.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_sqrt.h"

constexpr int ulps_single = 3;
constexpr int ulps_double = 0;

TEST_CASE("xtd::sqrt", "[sqrt][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::sqrt(float)") {
            validate<float, float, xtd::sqrt, mpfr_sqrtf>(platform, device, ulps_single);
          }

          SECTION("double xtd::sqrt(double)") {
            validate<double, double, xtd::sqrt, mpfr_sqrt>(platform, device, ulps_double);
          }

          SECTION("double xtd::sqrt(int)") {
            validate<double, int, xtd::sqrt, mpfr_sqrt>(platform, device, ulps_double);
          }

          SECTION("float xtd::sqrtf(float)") {
            validate<float, float, xtd::sqrtf, mpfr_sqrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sqrtf(double)") {
            validate<float, double, xtd::sqrtf, mpfr_sqrtf>(platform, device, ulps_single);
          }

          SECTION("float xtd::sqrtf(int)") {
            validate<float, int, xtd::sqrtf, mpfr_sqrtf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
