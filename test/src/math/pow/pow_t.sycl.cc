/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/pow.h"

// test headers
#include "common/sycl/device.h"
#include "common/sycl/platform.h"
#include "common/sycl/validate.h"
#include "mpfr_pow.h"

constexpr int ulps_single = 16;
constexpr int ulps_double = 16;

TEST_CASE("xtd::pow", "[pow][sycl]") {
  for (const auto &platform : test::sycl::platforms()) {
    DYNAMIC_SECTION("SYCL platform " << platform.index() << ": " << platform.name()) {
      for (const auto &device : platform.devices()) {
        DYNAMIC_SECTION("SYCL device " << platform.index() << '.' << device.index() << ": " << device.name()) {
          SECTION("float xtd::pow(float, float)") {
            test::sycl::validate<float, float, xtd::pow, mpfr_powf>(platform, device, ulps_single);
          }

          SECTION("double xtd::pow(double, double)") {
            test::sycl::validate<double, double, xtd::pow, mpfr_pow>(platform, device, ulps_double);
          }

          SECTION("double xtd::pow(int, int)") {
            test::sycl::validate<double, int, xtd::pow, mpfr_pow>(platform, device, ulps_double);
          }

          SECTION("float xtd::powf(float, float)") {
            test::sycl::validate<float, float, xtd::powf, mpfr_powf>(platform, device, ulps_single);
          }

          SECTION("float xtd::powf(double, double)") {
            test::sycl::validate<float, double, xtd::powf, mpfr_powf>(platform, device, ulps_single);
          }

          SECTION("float xtd::powf(int, int)") {
            test::sycl::validate<float, int, xtd::powf, mpfr_powf>(platform, device, ulps_single);
          }
        }
      }
    }
  }
}
