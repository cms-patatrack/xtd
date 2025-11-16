/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <span>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// SYCL headers
#include <sycl/sycl.hpp>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/math_inputs.h"
#include "common/sycl/device.h"
#include "common/sycl/platform.h"

namespace test::sycl {

  class Inputs {
  public:
    Inputs(const Device& device) : queue_{device.queue()}, size_(generate_input_values().size()) {
    }

    ~Inputs() {
      for (auto [key, ptr] : values_h_) {
        ::sycl::free(ptr, queue_);
      }
      for (auto [key, ptr] : values_d_) {
        ::sycl::free(ptr, queue_);
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
        T* data = ::sycl::malloc_host<T>(size_, queue_);
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
        // Copy the input dataset from the host to the device memory.
        // The copy can run asynchronously, becuse the lifetime of the input data is guaranteed
        // to be the same as that of the object, and future operations should use the same queue.
        std::span<const T> input = values_h<T>();
        T* data = ::sycl::malloc_device<T>(size_, queue_);
        queue_.memcpy(data, input.data(), size_ * sizeof(T));
        values_d_[typeid(T)] = (void*)data;
      }
      return std::span<const T>(static_cast<T*>(values_d_[typeid(T)]), size_);
    }

  private:
    // The SYCL queue is managed outside of this object, and is thread-safe.
    mutable ::sycl::queue queue_;

    // Input data.
    size_t size_;
    mutable std::unordered_map<std::type_index, void*> values_h_;
    mutable std::unordered_map<std::type_index, void*> values_d_;
  };

  // FIXME: this function is not thread safe.
  // Implement a reader/writer lock using std::shared_mutex ?
  inline const Inputs& inputs(const Platform& platform, const Device& device) {
    static std::unordered_map<size_t, Inputs> inputs;

    int index = (platform.index() << 16) + device.index();
    if (not inputs.contains(index)) {
      inputs.emplace(index, device);
    }
    return inputs.at(index);
  }

}  // namespace test::sycl
