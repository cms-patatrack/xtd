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

// CUDA headers
#include <cuda_runtime.h>

// mpfr::real headers
#include <real.hpp>

// xtd headers
#include "xtd/math/floor.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/cuda_version.h"
#include "common/math_inputs.h"

constexpr int ulps_single = 0;
constexpr int ulps_double = 0;

constexpr auto ref_function = [](mpfr_double x) { return mpfr::floor(x); };
constexpr auto ref_functionf = [](mpfr_single x) { return mpfr::floor(x); };

TEST_CASE("xtd::floor", "[floor][cuda]") {
  std::vector<double> values = generate_input_values();

  DYNAMIC_SECTION("CUDA platform: " << cuda_version()) {
    int deviceCount;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    for (int device = 0; device < deviceCount; ++device) {
      cudaDeviceProp properties;
      CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
      DYNAMIC_SECTION("CUDA device " << device << ": " << properties.name) {
        // set the current GPU
        CUDA_CHECK(cudaSetDevice(device));

        // create a CUDA stream for all the asynchronous operations on this GPU
        cudaStream_t queue;
        CUDA_CHECK(cudaStreamCreate(&queue));

        SECTION("float xtd::floor(float)") {
          test_a<float, float, xtd::floor, ref_function>(queue, values, ulps_single);
        }

        SECTION("double xtd::floor(double)") {
          test_a<double, double, xtd::floor, ref_function>(queue, values, ulps_double);
        }

        SECTION("double xtd::floor(int)") {
          test_a<double, int, xtd::floor, ref_function>(queue, values, ulps_double);
        }

        SECTION("float xtd::floorf(float)") {
          test_f<float, float, xtd::floorf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::floorf(double)") {
          test_f<float, double, xtd::floorf, ref_functionf>(queue, values, ulps_single);
        }

        SECTION("float xtd::floorf(int)") {
          test_f<float, int, xtd::floorf, ref_functionf>(queue, values, ulps_single);
        }

        CUDA_CHECK(cudaStreamDestroy(queue));
      }
    }
  }
}
