/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atan.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_atan.h"

constexpr int ulps_single = 5;
constexpr int ulps_double = 5;

TEST_CASE("xtd::atan", "[atan][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::atan(float)") {
            validate<float, float, xtd::atan, mpfr_atanf>(platform, device, ulps_single);
          }

          SECTION("double xtd::atan(double)") {
            validate<double, double, xtd::atan, mpfr_atan>(platform, device, ulps_double);
          }

          SECTION("double xtd::atan(int)") {
            validate<double, int, xtd::atan, mpfr_atan>(platform, device, ulps_double);
          }

          SECTION("float xtd::atanf(float)") {
            validate<float, float, xtd::atanf, mpfr_atanf>(platform, device, ulps_single);
          }

          SECTION("float xtd::atanf(double)") {
            validate<float, double, xtd::atanf, mpfr_atanf>(platform, device, ulps_single);
          }

          SECTION("float xtd::atanf(int)") {
            validate<float, int, xtd::atanf, mpfr_atanf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
