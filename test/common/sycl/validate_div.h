/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// C++ standard headers
#include <iomanip>
#include <iostream>
#include <type_traits>
#include <vector>

// SYCL headers
#include <sycl/sycl.hpp>

// xtd headers
#include <xtd/algorithm.h>

// test headers
#include "common/compare.h"
#include "common/halton.h"
#include "common/mpfr.h"
#include "common/sycl/device.h"
#include "common/sycl/inputs.h"
#include "common/sycl/platform.h"

namespace test::sycl {

  template <typename T>
  concept DivResultType = requires(T t) {
    // C struct
    requires std::is_standard_layout_v<T>;

    // quot data member
    t.quot;
    requires std::integral<decltype(T::quot)>;

    // rem data member
    t.rem;
    requires std::integral<decltype(T::rem)>;

    // members type
    requires std::same_as<decltype(T::quot), decltype(T::rem)>;
  };

  template <typename T>
  struct detailed {
    constexpr detailed(T value) : value_{value} {
    }

    T value_;
  };

  template <std::integral T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << val.value_;
    return out;
  }

  template <DivResultType T>
  std::ostream& operator<<(std::ostream& out, detailed<T> const& val) {
    out << '[' << val.value_.quot << ", " << val.value_.rem << ']';
    return out;
  }

  template <DivResultType ResultType,
            std::integral InputType,
            ResultType (*XtdFunc)(InputType, InputType),
            ResultType (*RefFunc)(InputType, InputType)>
  inline void validate_div(const Platform& platform, const Device& device) try {
    if constexpr (std::is_same_v<InputType, double>) {
      if (not device.device().has(::sycl::aspect::fp64)) {
        INFO("The device does not support double precision floating point operations, the test will be skipped.");
        return;
      }
    }

    const Inputs& input = inputs(platform, device);
    unsigned int size = input.size();
    std::span<const InputType> values_h = input.values_h<InputType>();
    std::span<const InputType> values_d = input.values_d<InputType>();

    // Allocate memory for the results and fill it with zeroes.
    std::vector<ResultType> result_h(size, ResultType{0, 0});
    ResultType* result_d = ::sycl::malloc_device<ResultType>(size, device.queue());
    device.queue().fill(result_d, ResultType{0, 0}, size);

    // Execute the xtd function on the SYCL device.
    device.queue().submit([&](::sycl::handler& cgh) {
      cgh.parallel_for(::sycl::range<1>(size), [=](::sycl::id<1> t) {
        // Generate a low-discrepancy deterministic sequence over [0, size)×[0, size).
        auto [i, j] = halton<2>(static_cast<size_t>(t), size);
        InputType x = values_d[i];
        InputType y = values_d[j];
        result_d[t] = static_cast<ResultType>(XtdFunc(x, y));
      });
    });

    // Copy the results back to the host and free the device memory.
    device.queue().copy(result_d, result_h.data(), size);
    device.queue().wait();
    ::sycl::free(result_d, device.queue());

    for (unsigned int t = 0; t < size; ++t) {
      // generate a low-discrepancy deterministic sequence over [0, size)×[0, size)
      auto [i, j] = halton<2>(t, size);
      InputType x = values_h[i];
      InputType y = values_h[j];
      // read the result of the xtd function
      ResultType result = result_h[t];
      // compute the reference
      ResultType reference = RefFunc(x, y);
      // log the comparison
      INFO("index: " << t << "\ninput: " << detailed(x) << ", " << detailed(y) << "\nxtd result: " << detailed(result)
                     << "\nreference:  " << detailed(reference) << '\n');
      // compare the result with the reference
      compare(result.quot, reference.quot);
      compare(result.rem, reference.rem);
    }
  } catch (::sycl::exception const& e) {
    std::cerr << "SYCL exception:\n"
              << e.what() << "\ncaught while running on platform "
              << platform.platform().get_info<::sycl::info::platform::name>() << ", device "
              << device.device().get_info<::sycl::info::device::name>() << '\n';
    std::exit(EXIT_FAILURE);
  }

}  // namespace test::sycl
