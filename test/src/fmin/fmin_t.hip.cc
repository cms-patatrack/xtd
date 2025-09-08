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
#include "xtd/math/fmin.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/hip_version.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 0;
constexpr int ulps_double = 0;

constexpr auto ref_fmin = [](mpfr_double y, mpfr_double x) { return mpfr::fmin(y, x); };
constexpr auto ref_fminf = [](mpfr_single y, mpfr_single x) { return mpfr::fmin(y, x); };

TEST_CASE("xtd::fmin", "[fmin][hip]") {
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

        SECTION("float xtd::fmin(float, float)") {
          test_2<float, float, xtd::fmin, ref_fmin>(queue, values, ulps_float);
        }

        SECTION("double xtd::fmin(double, double)") {
          test_2<double, double, xtd::fmin, ref_fmin>(queue, values, ulps_double);
        }

        SECTION("double xtd::fmin(int, int)") {
          test_2<double, int, xtd::fmin, ref_fmin>(queue, values, ulps_double);
        }

        SECTION("float xtd::fminf(float, float)") {
          test_2f<float, float, xtd::fminf, ref_fminf>(queue, values, ulps_float);
        }

        SECTION("float xtd::fminf(double, double)") {
          test_2f<float, double, xtd::fminf, ref_fminf>(queue, values, ulps_float);
        }

        SECTION("float xtd::fminf(int, int)") {
          test_2f<float, int, xtd::fminf, ref_fminf>(queue, values, ulps_float);
        }

        HIP_CHECK(hipStreamDestroy(queue));
      }
    }
  }
}
