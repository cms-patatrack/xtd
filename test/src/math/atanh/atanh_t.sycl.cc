/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/atanh.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_atanh.h"

constexpr int ulps_single = 5;
constexpr int ulps_double = 5;

TEST_CASE("xtd::atanh", "[atanh][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::atanh(float)") {
            validate<float, float, xtd::atanh, mpfr_atanhf>(platform, device, ulps_single);
          }

          SECTION("double xtd::atanh(double)") {
            validate<double, double, xtd::atanh, mpfr_atanh>(platform, device, ulps_double);
          }

          SECTION("double xtd::atanh(int)") {
            validate<double, int, xtd::atanh, mpfr_atanh>(platform, device, ulps_double);
          }

          SECTION("float xtd::atanhf(float)") {
            validate<float, float, xtd::atanhf, mpfr_atanhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::atanhf(double)") {
            validate<float, double, xtd::atanhf, mpfr_atanhf>(platform, device, ulps_single);
          }

          SECTION("float xtd::atanhf(int)") {
            validate<float, int, xtd::atanhf, mpfr_atanhf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
