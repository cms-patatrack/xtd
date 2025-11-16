/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <cstdlib>
#include <span>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>

// xtd headers
#include <xtd/concepts/arithmetic.h>

// test headers
#include "common/math_inputs.h"

namespace test::cpu {

  class Inputs {
  public:
    Inputs() : size_(generate_input_values().size()) {
    }

    ~Inputs() {
      for (auto [key, ptr] : values_) {
        std::free(ptr);
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
    std::span<const T> values() const {
      if (not values_.contains(typeid(T))) {
        // Generate the input dataset in double precision, and convert it to the required type.
        const std::vector<double>& input = generate_input_values();
        T* data = static_cast<T*>(std::malloc(size_ * sizeof(T)));
        for (size_t i = 0; i < size_; ++i) {
          data[i] = static_cast<T>(input[i]);
        }
        values_[typeid(T)] = static_cast<void*>(data);
      }
      return std::span<const T>(static_cast<T*>(values_[typeid(T)]), size_);
    }

  private:
    // Input data.
    size_t size_;
    mutable std::unordered_map<std::type_index, void*> values_;
  };

  inline const Inputs& inputs() {
    static Inputs inputs;
    return inputs;
  }

}  // namespace test::cpu
