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
#include "xtd/math/fmod.h"

// test headers
#include "common/hip_check.h"
#include "common/hip_test.h"
#include "common/hip_version.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x, mpfr_double y) -> mpfr_double { return mpfr::fmod(x, y); };
constexpr auto ref_functionf = [](mpfr_single x, mpfr_single y) -> mpfr_single { return mpfr::fmod(x, y); };

TEST_CASE("xtd::fmod", "[fmod][hip]") {
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

        SECTION("float xtd::fmod(float, float)") {
          test_aa<float, float, xtd::fmod, ref_function>(queue, values, ulps_single);
        }

        SECTION("double xtd::fmod(double, double)") {
          test_aa<double, double, xtd::fmod, ref_function>(queue, values, ulps_double);
        }

        SECTION("double xtd::fmod(int, int)") {
          test_aa<double, int, xtd::fmod, ref_function>(queue, values, ulps_double);
        }

        SECTION("float xtd::fmodf(float, float)") {
          test_ff<float, float, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::fmodf(double, double)") {
          test_ff<float, double, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::fmodf(int, int)") {
          test_ff<float, int, xtd::fmodf, ref_functionf>(queue, values, ulps_single);
        }

        HIP_CHECK(hipStreamDestroy(queue));
      }
    }
  }
}
