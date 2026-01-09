/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string_view>

// CUDA headers
#include <cuda_runtime.h>

// test headers
#include "common/cuda/cuda_check.h"

namespace test::cuda {

  class Device {
  public:
    Device(int id) : index_(id) {
      // Read the GPU properties.
      CUDA_CHECK(cudaGetDeviceProperties(&properties_, index_));

      // Set the current GPU.
      CUDA_CHECK(cudaSetDevice(index_));

      // Create a CUDA stream for all the asynchronous operations on this GPU.
      CUDA_CHECK(cudaStreamCreate(&queue_));
    }

    ~Device() {
      CUDA_CHECK(cudaStreamSynchronize(queue_));
      CUDA_CHECK(cudaStreamDestroy(queue_));
      CUDA_CHECK(cudaSetDevice(index_));
      CUDA_CHECK(cudaDeviceReset());
    }

    Device(const Device&) = delete;
    Device& operator=(const Device&) = delete;

    Device(Device&&) = default;
    Device& operator=(Device&&) = default;

    int index() const {
      return index_;
    }

    std::string_view name() const {
      return properties_.name;
    }

    cudaStream_t queue() const {
      return queue_;
    }

  private:
    // Device properties.
    int index_;
    cudaDeviceProp properties_;
    cudaStream_t queue_;
  };

}  // namespace test::cuda
