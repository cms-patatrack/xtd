/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

// C++ standard headers
#include <string>
#include <vector>
using namespace std::literals;

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/trunc.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/hip_version.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

TEST_CASE("xtd::trunc", "[trunc][hip]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("HIP platform: " << hip_version()) {
    int deviceCount;
    HIP_CHECK(hipGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      hipDeviceProp_t properties;
      HIP_CHECK(hipGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("HIP device " << device << ": " << properties.name) {
        // set the current GPU
        HIP_CHECK(hipSetDevice(device));

        // create a HIP stream for all the asynchronous operations on this GPU
        hipStream_t queue;
        HIP_CHECK(hipStreamCreate(&queue));

        SECTION("float xtd::trunc(float)") {
          test<float, float, xtd::trunc, mpfr::trunc>(queue, values, ulps_float);
        }

        SECTION("double xtd::trunc(double)") {
          test<double, double, xtd::trunc, mpfr::trunc>(queue, values, ulps_double);
        }

        SECTION("double xtd::trunc(int)") {
          test<double, int, xtd::trunc, mpfr::trunc>(queue, values, ulps_double);
        }

        SECTION("float xtd::truncf(float)") {
          test_f<float, float, xtd::truncf, mpfr::trunc>(queue, values, ulps_float);
        }

        SECTION("float xtd::truncf(double)") {
          test_f<float, double, xtd::truncf, mpfr::trunc>(queue, values, ulps_float);
        }

        SECTION("float xtd::truncf(int)") {
          test_f<float, int, xtd::truncf, mpfr::trunc>(queue, values, ulps_float);
        }

        HIP_CHECK(hipStreamDestroy(queue));
      }
    }
  }
}
