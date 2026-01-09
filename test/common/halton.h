/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

// Implement Halton sequences up to 20 dimensions.
// See https://en.wikipedia.org/wiki/Halton_sequence.

// C++ standard headers
#include <algorithm>
#include <array>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <iostream>
#include <utility>

constexpr inline uint32_t primes[] = {2u,  3u,  5u,  7u,  11u, 13u, 17u, 19u, 23u, 29u,
                                      31u, 37u, 41u, 43u, 47u, 53u, 59u, 61u, 67u, 71u};

// Returns a floating point number in the semi-open range [0., 1.)
template <std::floating_point F = float, std::integral I>
constexpr inline F halton_unit(I index, std::type_identity_t<I> base) {
  F f = 1.;
  F r = 0.;
  while (index > 0) {
    f /= base;
    r += f * (index % base);
    index /= base;
  }
  return r;
}

// Returns an integer number in the semi-open range [0, range)
template <std::floating_point F = float, std::integral I, std::integral T>
constexpr inline I halton_base(I index, std::type_identity_t<I> base, T range) {
  F x = halton_unit<F, I>(index, base);
  return std::min<T>(static_cast<T>(std::round(x * range)), range - 1);
}

// Returns an integer number in the semi-open range [0, range)
template <std::floating_point F = float, std::integral I, std::integral T>
constexpr inline I halton(I index, T range) {
  constexpr I base = 2u;
  F x = halton_unit<F, I>(index, base);
  return std::min<T>(static_cast<T>(std::round(x * range)), range - 1);
}

// Helper function to unroll the loop at compile time
template <typename T, typename F, typename I, std::size_t... Is>
constexpr std::array<T, sizeof...(Is)> __halton_impl(I index, I range, std::index_sequence<Is...>) {
  return {([&] {
    const I base = primes[Is];
    // skip the first N points to improve the decorrelation
    F x = halton_unit<F, I>(index + sizeof...(Is), base);
    return std::min<T>(static_cast<T>(std::round(x * range)), range - 1);
  }())...};
}

// Returns an N-dimensional array of integers in the semi-open range [0, range) * [0, range) * ...
template <uint32_t N, std::floating_point F = float, std::integral I, std::integral T>
constexpr inline std::array<T, N> halton(I index, T range) {
  static_assert(N < std::size(primes), "halton() is implemented only up to N dimensions");
  return __halton_impl<T, F, I>(index, range, std::make_index_sequence<N>{});
}
