/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <optional>
#include <span>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// CUDA headers
#include <cuda_runtime.h>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/cuda/cuda_check.h"
#include "common/cuda/device.h"
#include "common/math_inputs.h"

namespace test::cuda {

  class Inputs {
  public:
    Inputs(cudaStream_t queue) : queue_(queue), size_(generate_input_values().size()) {
    }

    ~Inputs() {
      for (auto [key, ptr] : values_h_) {
        CUDA_CHECK(cudaFreeHost(ptr));
      }
      for (auto [key, ptr] : values_d_) {
        CUDA_CHECK(cudaFreeAsync(ptr, queue_));
      }
    }

    // Not copyable.
    Inputs(const Inputs&) = delete;
    Inputs& operator=(const Inputs&) = delete;

    // Movable.
    Inputs(Inputs&&) = default;
    Inputs& operator=(Inputs&&) = default;

    size_t size() const {
      return size_;
    }

    // FIXME: this function is not thread safe.
    // Implement a reader/writer lock using std::shared_mutex ?
    template <xtd::arithmetic T>
    std::span<const T> values_h() const {
      if (not values_h_.contains(typeid(T))) {
        // Generate the input dataset in double precision, and convert it to the required type.
        const std::vector<double>& input = generate_input_values();
        T* data = nullptr;
        CUDA_CHECK(cudaMallocHost(&data, size_ * sizeof(T)));
        for (size_t i = 0; i < size_; ++i) {
          data[i] = static_cast<T>(input[i]);
        }
        values_h_[typeid(T)] = static_cast<void*>(data);
      }
      return std::span<const T>(static_cast<T*>(values_h_[typeid(T)]), size_);
    }

    // FIXME: this function is not thread safe.
    // Implement a reader/writer lock using std::shared_mutex ?
    template <xtd::arithmetic T>
    std::span<const T> values_d() const {
      if (not values_d_.contains(typeid(T))) {
        // Copy the input dataset from the host to the GPU memory.
        // The copy can run asynchronously, becuse the lifetime of the input data is guaranteed
        // to be the same as that of the object, and future operations should use the same stream.
        std::span<const T> input = values_h<T>();
        T* data = nullptr;
        CUDA_CHECK(cudaMallocAsync(&data, size_ * sizeof(T), queue_));
        CUDA_CHECK(cudaMemcpyAsync(data, input.data(), size_ * sizeof(T), cudaMemcpyHostToDevice, queue_));
        values_d_[typeid(T)] = (void*)data;
      }
      return std::span<const T>(static_cast<T*>(values_d_[typeid(T)]), size_);
    }

  private:
    // The CUDA stream is managed outside of this object.
    cudaStream_t queue_;

    // Input data.
    size_t size_;
    mutable std::unordered_map<std::type_index, void*> values_h_;
    mutable std::unordered_map<std::type_index, void*> values_d_;
  };

  inline std::vector<std::optional<Inputs>> inputs_impl() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    std::vector<std::optional<Inputs>> inputs(count);
    return inputs;
  }

  // FIXME: this function is not thread safe.
  // Implement a reader/writer lock using std::shared_mutex ?
  inline const Inputs& inputs(const Device& device) {
    static std::vector<std::optional<Inputs>> inputs = inputs_impl();

    if (not inputs[device.index()]) {
      inputs[device.index()].emplace(device.queue());
    }
    return inputs[device.index()].value();
  }

}  // namespace test::cuda
