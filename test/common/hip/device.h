/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <string_view>

// HIP headers
#include <hip/hip_runtime.h>

// test headers
#include "common/hip/hip_check.h"

namespace test::hip {

  class Device {
  public:
    Device(int id) : index_(id) {
      // Read the GPU properties.
      HIP_CHECK(hipGetDeviceProperties(&properties_, index_));

      // Set the current GPU.
      HIP_CHECK(hipSetDevice(index_));

      // Create a HIP stream for all the asynchronous operations on this GPU.
      HIP_CHECK(hipStreamCreate(&queue_));
    }

    ~Device() {
      HIP_CHECK(hipStreamSynchronize(queue_));
      HIP_CHECK(hipStreamDestroy(queue_));
      HIP_CHECK(hipSetDevice(index_));
      HIP_CHECK(hipDeviceReset());
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

    hipStream_t queue() const {
      return queue_;
    }

  private:
    // Device properties.
    int index_;
    hipDeviceProp_t properties_;
    hipStream_t queue_;
  };

}  // namespace test::hip
