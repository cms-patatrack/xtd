/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// xtd headers
#include "xtd/math/tanh.h"

// test headers
#include "common/cuda/platform.h"
#include "common/cuda/validate.h"
#include "mpfr_tanh.h"

constexpr int ulps_single = 2;
constexpr int ulps_double = 1;

TEST_CASE("xtd::tanh", "[tanh][cuda]") {
  const auto& platform = test::cuda::platform();
  DYNAMIC_SECTION("CUDA platform: " << platform.name()) {
    for (const auto& device : platform.devices()) {
      DYNAMIC_SECTION("CUDA device " << device.index() << ": " << device.name()) {
        SECTION("float xtd::tanh(float)") {
          validate<float, float, xtd::tanh, mpfr_tanhf>(device, ulps_single);
        }

        SECTION("double xtd::tanh(double)") {
          validate<double, double, xtd::tanh, mpfr_tanh>(device, ulps_double);
        }

        SECTION("double xtd::tanh(int)") {
          validate<double, int, xtd::tanh, mpfr_tanh>(device, ulps_double);
        }

        SECTION("float xtd::tanhf(float)") {
          validate<float, float, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
        }

        SECTION("float xtd::tanhf(double)") {
          validate<float, double, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
        }

        SECTION("float xtd::tanhf(int)") {
          validate<float, int, xtd::tanhf, mpfr_tanhf>(device, ulps_single);
        }
      }
    }
  }
}
