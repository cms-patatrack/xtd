/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/acos.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_acos.h"

constexpr int ulps_single = 4;
constexpr int ulps_double = 4;

TEST_CASE("xtd::acos", "[acos][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::acos(float)") {
            validate<float, float, xtd::acos, mpfr_acosf>(platform, device, ulps_single);
          }

          SECTION("double xtd::acos(double)") {
            validate<double, double, xtd::acos, mpfr_acos>(platform, device, ulps_double);
          }

          SECTION("double xtd::acos(int)") {
            validate<double, int, xtd::acos, mpfr_acos>(platform, device, ulps_double);
          }

          SECTION("float xtd::acosf(float)") {
            validate<float, float, xtd::acosf, mpfr_acosf>(platform, device, ulps_single);
          }

          SECTION("float xtd::acosf(double)") {
            validate<float, double, xtd::acosf, mpfr_acosf>(platform, device, ulps_single);
          }

          SECTION("float xtd::acosf(int)") {
            validate<float, int, xtd::acosf, mpfr_acosf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
