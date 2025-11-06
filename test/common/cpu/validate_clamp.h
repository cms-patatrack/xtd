/*
 * Copyright 2025 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <concepts>
#include <iomanip>
#include <iostream>
#include <span>

// test headers
#include "common/compare.h"
#include "common/cpu/inputs.h"
#include "common/halton.h"

namespace test::cpu {

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::floating_point T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    std::ostringstream buffer;
    buffer << std::fixed << std::setprecision(std::numeric_limits<T>::max_digits10) << val.value_ << " ["
           << std::hexfloat << val.value_ << "]";
    out << buffer.str();
    return out;
  }

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <std::floating_point ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate_clamp(const Device& device, int ulps = 0) {
    std::span<const InputType> values = inputs().values<InputType>();
    unsigned int size = values.size();
    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size)
      auto [i, j, k] = halton<3>(t, size);
      InputType x = values[i];
      InputType y = std::min(values[j], values[k]);
      InputType z = std::max(values[j], values[k]);
      // compute the result of the xtd function
      ResultType result = XtdFunc(x, y, z);
      // compute the reference
      ResultType reference = RefFunc(x, y, z);
      // log the comparison
      INFO("input: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference, ulps);
    }
  }

  template <std::integral ResultType,
            typename InputType,
            ResultType (*XtdFunc)(InputType, InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType, InputType)>
  inline void validate_clamp(const Device& device) {
    std::span<const InputType> values = inputs().values<InputType>();
    unsigned int size = values.size();
    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)×[0, size)
      auto [i, j, k] = halton<3>(t, size);
      InputType x = values[i];
      InputType y = std::min(values[j], values[k]);
      InputType z = std::max(values[j], values[k]);
      // compute the result of the xtd function
      ResultType result = XtdFunc(x, y, z);
      // compute the reference
      ResultType reference = RefFunc(x, y, z);
      // log the comparison
      INFO("input: " << detailed(x) << ", " << detailed(y) << ", " << detailed(z)
                     << "\nxtd result: " << detailed(result) << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result, reference);
    }
  }

}  // namespace test::cpu
