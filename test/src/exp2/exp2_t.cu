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
#include "xtd/math/exp2.h"

// test headers
#include "common/cuda_check.h"
#include "common/cuda_test.h"
#include "common/math_inputs.h"

constexpr int ulps_float = 2;
constexpr int ulps_double = 1;

TEST_CASE("xtd::exp2", "[exp2][cuda]") {
  std::vector<double> values = generate_input_values();

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

      SECTION("float xtd::exp2(float)") {
        test<float, float, xtd::exp2, mpfr::exp2>(queue, values, ulps_float);
      }

      SECTION("double xtd::exp2(double)") {
        test<double, double, xtd::exp2, mpfr::exp2>(queue, values, ulps_double);
      }

      SECTION("double xtd::exp2(int)") {
        test<double, int, xtd::exp2, mpfr::exp2>(queue, values, ulps_double);
      }

      SECTION("float xtd::exp2f(float)") {
        test_f<float, float, xtd::exp2f, mpfr::exp2>(queue, values, ulps_float);
      }

      SECTION("float xtd::exp2f(double)") {
        test_f<float, double, xtd::exp2f, mpfr::exp2>(queue, values, ulps_float);
      }

      SECTION("float xtd::exp2f(int)") {
        test_f<float, int, xtd::exp2f, mpfr::exp2>(queue, values, ulps_float);
      }

      CUDA_CHECK(cudaStreamDestroy(queue));
    }
  }
}
