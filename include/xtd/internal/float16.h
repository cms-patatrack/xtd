/*
 * Copyright 2026 European Organization for Nuclear Research (CERN)
 * Authors: Andrea Bocci <andrea.bocci@cern.ch>, Aurora Perego <aurora.perego@cern.ch>, Simone Balducci <simone.balducci@cern.ch>, Patrick Kopper <patrick.kopper@cern.ch>
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <concepts>
#include <cmath>

#include <cstdint>   // For uint16_t
#include <bit>       // For std::bit_cast
#include <ostream>

#include "xtd/internal/defines.h"

#if defined(XTD_TARGET_CUDA)
#include <cuda_fp16.h>
#elif defined(XTD_TARGET_HIP)
#include <hip/hip_fp16.h>
#if defined(__HIP_PLATFORM_AMD__) && !defined(__double2half)
__device__ inline __half __double2half(double d) {
    return static_cast<__half>(d);
}
#endif
#elif defined(XTD_TARGET_SYCL)
#include <sycl/sycl.hpp>
#endif

namespace xtd {
  class float16_t {
  public:
      constexpr float16_t() noexcept = default;
      constexpr float16_t(const float16_t&) noexcept = default;
      constexpr float16_t(float16_t&&) noexcept = default;
      constexpr float16_t& operator=(const float16_t&) noexcept = default;
      constexpr float16_t& operator=(float16_t&&) noexcept = default;
      ~float16_t() noexcept = default;

      /* ========================================================================
       * CONSTRUCTOR
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline float16_t(float f) noexcept
#if defined(XTD_TARGET_CUDA)
          // CUDA device code
          : bits_(__half_as_ushort(__float2half(f))) {}
#elif defined(XTD_TARGET_HIP)
          // HIP/ROCm device code
          : bits_(__half_as_ushort(__float2half(f))) {}
#elif defined(XTD_TARGET_SYCL)
          // SYCL device code
          : bits_(sycl::bit_cast<uint16_t>(static_cast<sycl::half>(f))) {}
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          // standard C/C++ code
          : bits_(std::bit_cast<uint16_t>(static_cast<std::float16_t>(f))) {}
#else
          // c++20 and below fallback
          : bits_(float_to_half(std::bit_cast<std::uint32_t>(f))) {}
#endif
#endif

      XTD_DEVICE_FUNCTION constexpr inline float16_t(double d) noexcept
#if defined(XTD_TARGET_CUDA)
          // CUDA device code
          : bits_(__half_as_ushort(__double2half(d))) {}
#elif defined(XTD_TARGET_HIP)
          // HIP/ROCm device code
          : bits_(__half_as_ushort(__double2half(d))) {}
#elif defined(XTD_TARGET_SYCL)
          // SYCL device code
          : bits_(sycl::bit_cast<uint16_t>(static_cast<sycl::half>(d))) {}
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          // standard C/C++ code
          : bits_(std::bit_cast<uint16_t>(static_cast<std::float16_t>(d))) {}
#elif defined(__FLT16_MANT_DIG__)
          : bits_(std::bit_cast<uint16_t>(static_cast<_Float16>(d))) {}
#else
          // c++20 and below fallback
          : bits_(double_to_half(std::bit_cast<std::uint64_t>(d))) {}
#endif
#endif

      template <typename T>
          requires std::convertible_to<T, float> && (!std::same_as<std::decay_t<T>, float>) && (!std::same_as<std::decay_t<T>, float16_t>)
      XTD_DEVICE_FUNCTION constexpr inline float16_t(T val) noexcept
          : float16_t(static_cast<float>(val)) {}

      /* ========================================================================
       * CONVERTOR
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline operator float() const noexcept {
#if defined(XTD_TARGET_CUDA)
          // CUDA device code
          return __half2float(__ushort_as_half(bits_));
#elif defined(XTD_TARGET_HIP)
          // HIP/ROCm device code
          return __half2float(__ushort_as_half(bits_));
#elif defined(XTD_TARGET_SYCL)
          // SYCL device code
          return static_cast<float>(sycl::bit_cast<sycl::half>(bits_));
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          // standard C/C++ code
          return static_cast<float>(std::bit_cast<std::float16_t>(bits_));
#else
          // c++20 and below fallback
          return std::bit_cast<float>(half_to_float(bits_));
#endif
#endif
      }

      /* ========================================================================
       * COMPATIBILITY CONSTRUCTOR FROM VENDOR
       * ===================================================================== */
      // NON-EXPLICIT constructor allows implicit conversion from std::float16_t (_Float16)
#if defined(XTD_TARGET_CUDA)
      // CUDA device code
      XTD_DEVICE_FUNCTION constexpr inline float16_t(::__half f) noexcept
          : bits_(__half_as_ushort(f)) {}
#elif defined(XTD_TARGET_HIP)
      // HIP/ROCm device code
      XTD_DEVICE_FUNCTION constexpr inline float16_t(::__half f) noexcept
          : bits_(__half_as_ushort(f)) {}
#elif defined(XTD_TARGET_SYCL)
      // SYCL device code
      XTD_DEVICE_FUNCTION constexpr inline float16_t(sycl::half f) noexcept
          : bits_(sycl::bit_cast<std::uint16_t>(f)) {}
#endif

#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
      // C++23 standard float16_t
      XTD_DEVICE_FUNCTION constexpr inline float16_t(std::float16_t f) noexcept
          : bits_(std::bit_cast<std::uint16_t>(f)) {}
#elif defined(__FLT16_MANT_DIG__) && !defined(__NVCC__)
      // Core C23 / GNU extension native _Float16
      XTD_DEVICE_FUNCTION constexpr inline float16_t(_Float16 f) noexcept
          : bits_(std::bit_cast<std::uint16_t>(f)) {}
#endif

      /* ========================================================================
       * COMPATIBILITY CONVERTER TO VENDOR
       * ===================================================================== */
#if defined(XTD_TARGET_CUDA)
      // CUDA device code
      XTD_DEVICE_FUNCTION constexpr inline operator ::__half() const noexcept {
          return __ushort_as_half(bits_);
      }
#elif defined(XTD_TARGET_HIP)
      // HIP/ROCm device code
      XTD_DEVICE_FUNCTION constexpr inline operator ::__half() const noexcept {
          return __ushort_as_half(bits_);
      }
#elif defined(XTD_TARGET_SYCL)
      // SYCL device code
      XTD_DEVICE_FUNCTION constexpr inline operator sycl::half() const noexcept {
          return sycl::bit_cast<sycl::half>(bits_);
      }
#endif
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
      // standard C/C++ code
      XTD_DEVICE_FUNCTION constexpr inline operator std::float16_t() const noexcept {
          return std::bit_cast<std::float16_t>(bits_);
      }
#endif

      /* ========================================================================
       * UNARY OPERATORS
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline float16_t operator+() const noexcept { return *this; }
      XTD_DEVICE_FUNCTION constexpr inline float16_t operator-() const noexcept {
          float16_t res = *this;
          res.bits_ ^= 0x8000;
          return res;
      }

  /* ========================================================================
       * ARITHMETIC OPERATORS
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline float16_t operator+(float16_t rhs) const noexcept {
#if defined(XTD_TARGET_CUDA)
          return __hadd(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_HIP)
          return __hadd(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_SYCL)
          return static_cast<sycl::half>(*this) + static_cast<sycl::half>(rhs);
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          return static_cast<std::float16_t>(*this) + static_cast<std::float16_t>(rhs);
#else
          return static_cast<float>(*this) + static_cast<float>(rhs);
#endif
#endif
      }

      XTD_DEVICE_FUNCTION constexpr inline float16_t operator-(float16_t rhs) const noexcept {
#if defined(XTD_TARGET_CUDA)
          return __hsub(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_HIP)
          return __hsub(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_SYCL)
          return static_cast<sycl::half>(*this) - static_cast<sycl::half>(rhs);
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          return static_cast<std::float16_t>(*this) - static_cast<std::float16_t>(rhs);
#else
          return static_cast<float>(*this) - static_cast<float>(rhs);
#endif
#endif
      }

      XTD_DEVICE_FUNCTION constexpr inline float16_t operator*(float16_t rhs) const noexcept {
#if defined(XTD_TARGET_CUDA)
          return __hmul(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_HIP)
          return __hmul(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_SYCL)
          return static_cast<sycl::half>(*this) * static_cast<sycl::half>(rhs);
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          return static_cast<std::float16_t>(*this) * static_cast<std::float16_t>(rhs);
#else
          return static_cast<float>(*this) * static_cast<float>(rhs);
#endif
#endif
      }

      XTD_DEVICE_FUNCTION constexpr inline float16_t operator/(float16_t rhs) const noexcept {
#if defined(XTD_TARGET_CUDA)
          return __hdiv(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_HIP)
          return __hdiv(static_cast<::__half>(*this), static_cast<::__half>(rhs));
#elif defined(XTD_TARGET_SYCL)
          return static_cast<sycl::half>(*this) / static_cast<sycl::half>(rhs);
#else
#if defined(__STDCPP_FLOAT16_T__) || XTD_HAS_STDFLOAT16
          return static_cast<std::float16_t>(*this) / static_cast<std::float16_t>(rhs);
#else
          return static_cast<float>(*this) / static_cast<float>(rhs);
#endif
#endif
      }

      XTD_DEVICE_FUNCTION constexpr inline float16_t& operator+=(float16_t rhs) noexcept { *this = *this + rhs; return *this; }
      XTD_DEVICE_FUNCTION constexpr inline float16_t& operator-=(float16_t rhs) noexcept { *this = *this - rhs; return *this; }
      XTD_DEVICE_FUNCTION constexpr inline float16_t& operator*=(float16_t rhs) noexcept { *this = *this * rhs; return *this; }
      XTD_DEVICE_FUNCTION constexpr inline float16_t& operator/=(float16_t rhs) noexcept { *this = *this / rhs; return *this; }

      /* ========================================================================
       * TEMPLATED ARITHMETIC OPERATORS
       * ===================================================================== */
      template <typename L, typename R>
      requires (std::is_same_v<L, float16_t> && std::is_arithmetic_v<R>) ||
               (std::is_arithmetic_v<L> && std::is_same_v<R, float16_t>)
      XTD_DEVICE_FUNCTION friend constexpr inline float operator+(L lhs, R rhs) noexcept {
          return static_cast<float>(lhs) + static_cast<float>(rhs);
      }

      template <typename L, typename R>
      requires (std::is_same_v<L, float16_t> && std::is_arithmetic_v<R>) ||
               (std::is_arithmetic_v<L> && std::is_same_v<R, float16_t>)
      XTD_DEVICE_FUNCTION friend constexpr inline float operator*(L lhs, R rhs) noexcept {
          return static_cast<float>(lhs) * static_cast<float>(rhs);
      }

      template <typename L, typename R>
      requires (std::is_same_v<L, float16_t> && std::is_arithmetic_v<R>) ||
               (std::is_arithmetic_v<L> && std::is_same_v<R, float16_t>)
      XTD_DEVICE_FUNCTION friend constexpr inline float operator-(L lhs, R rhs) noexcept {
          return static_cast<float>(lhs) - static_cast<float>(rhs);
      }

      template <typename L, typename R>
      requires (std::is_same_v<L, float16_t> && std::is_arithmetic_v<R>) ||
               (std::is_arithmetic_v<L> && std::is_same_v<R, float16_t>)
      XTD_DEVICE_FUNCTION friend constexpr inline float operator/(L lhs, R rhs) noexcept {
          return static_cast<float>(lhs) / static_cast<float>(rhs);
      }

      /* ========================================================================
       * COMPARISON OPERATORS
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline bool operator==(const float16_t& rhs) const noexcept {
          if (std::isnan(static_cast<float>(*this)) || std::isnan(static_cast<float>(rhs))) {
              return false;
          }
          return to_ordered_int(bits_) == to_ordered_int(rhs.bits_);
      }
#if defined(XTD_TARGET_CUDA) || defined(XTD_TARGET_HIP) || defined(XTD_TARGET_SYCL)
      XTD_DEVICE_FUNCTION constexpr inline bool operator!=(const float16_t& rhs) const noexcept { return !(*this == rhs); }
      XTD_DEVICE_FUNCTION constexpr inline bool operator>(const float16_t& rhs) const noexcept { return rhs < *this; }
      XTD_DEVICE_FUNCTION constexpr inline bool operator<=(const float16_t& rhs) const noexcept { return !(rhs < *this); }
      XTD_DEVICE_FUNCTION constexpr inline bool operator>=(const float16_t& rhs) const noexcept { return !(*this < rhs); }
      XTD_DEVICE_FUNCTION constexpr inline bool operator<(const float16_t& rhs) const noexcept {
          if (std::isnan(static_cast<float>(*this)) || std::isnan(static_cast<float>(rhs))) {
              return false;
          }
          return to_ordered_int(bits_) < to_ordered_int(rhs.bits_);
      }
#else
      XTD_DEVICE_FUNCTION constexpr std::partial_ordering operator<=>(const float16_t& rhs) const noexcept {
          if (std::isnan(static_cast<float>(*this)) || std::isnan(static_cast<float>(rhs))) {
              return std::partial_ordering::unordered;
          }
          return to_ordered_int(bits_) <=> to_ordered_int(rhs.bits_);
      }
#endif

      /* ========================================================================
       * BITWISE OPERATORS
       * ===================================================================== */
      XTD_DEVICE_FUNCTION constexpr inline float16_t operator&(uint16_t mask) const noexcept {
          float16_t res;
          res.bits_ = bits_ & mask;
          return res;
      }

      XTD_DEVICE_FUNCTION constexpr inline float16_t& operator&=(uint16_t mask) noexcept {
          bits_ &= mask;
          return *this;
      }

      /* ========================================================================
       * UTILITIES
       * ===================================================================== */
      friend std::ostream& operator<<(std::ostream& os, float16_t const& c)
      {
          return os << static_cast<float>(c);
      }

  private:
      // Needs initialization to be compatible with <C++20
      // uint16_t bits_{0};
      uint16_t bits_;

      XTD_DEVICE_FUNCTION constexpr static inline std::int16_t to_ordered_int(std::uint16_t u) noexcept {
          u = (u == 0x8000) ? 0 : u;  // Handling signed zero
          return (0x8000 & u) ? static_cast<std::int16_t>(0x8000 - u)
                              : static_cast<std::int16_t>(u);  // Converting sign-magnitude to two's complement
      }

      /* Copyright <2020> <Feng Wang>
       * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:
       * 1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
       * 2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.
       * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
       */
      XTD_DEVICE_FUNCTION constexpr inline static std::uint16_t float_to_half( const std::uint32_t f ) noexcept
      {
          static_assert(sizeof(float) == sizeof(uint32_t));

          const std::uint32_t f_s_mask = 0x80000000;
          const std::uint32_t f_e_mask = 0x7f800000;
          const std::uint32_t f_m_mask = 0x007fffff;
          const std::uint32_t f_m_hidden_bit = 0x00800000;
          const std::uint32_t f_snan_mask = 0x7fc00000;
          const std::uint32_t f_e_pos = 0x00000017;
          const std::uint32_t h_e_pos = 0x0000000a;
          const std::uint32_t h_e_mask = 0x00007c00;
          const std::uint32_t h_snan_mask = 0x00007e00;
          const std::uint32_t h_e_max_value = 0x0000001e;
          const std::uint32_t f_h_s_pos_offset = 0x00000010;
          const std::uint32_t f_h_bias_offset = 0x00000070;
          const std::uint32_t f_h_m_pos_offset = 0x0000000d;
          const std::uint32_t h_nan_min = 0x00007c01;
          const std::uint32_t f_h_e_biased_flag = 0x0000008e;

          const std::uint32_t f_s = ( f & f_s_mask );
          const std::uint32_t f_e = ( f & f_e_mask );
          const std::uint16_t h_s = ( f_s >> f_h_s_pos_offset );
          const std::uint32_t f_m = ( f & f_m_mask );
          const std::uint16_t f_e_amount = ( f_e >> f_e_pos );
          const std::uint32_t f_e_half_bias = ( f_e_amount - f_h_bias_offset );
          const std::uint32_t f_snan = ( f & f_snan_mask );

          // Denormal shift calculation capped to prevent shift overflow (round to nearest, ties to even)
          const std::uint32_t raw_denorm_sa = ( 1 - f_e_half_bias );
          const std::uint32_t is_sa_overflow_msb = ( 11 - raw_denorm_sa );
          const std::uint32_t denorm_sa_clamped = _uint32_sels( is_sa_overflow_msb, 11, raw_denorm_sa );

          const std::uint32_t f_m_with_hidden = ( f_m | f_m_hidden_bit );
          const std::uint32_t denorm_total_shift = f_h_m_pos_offset + denorm_sa_clamped;
          const std::uint32_t denorm_bias = ( 1u << ( denorm_total_shift - 1 ) ) - 1u;
          const std::uint32_t denorm_lsb = ( f_m_with_hidden >> denorm_total_shift ) & 1u;
          const std::uint32_t h_m_denorm_calc = ( f_m_with_hidden + denorm_bias + denorm_lsb ) >> denorm_total_shift;
          const std::uint32_t h_m_denorm = _uint32_sels( is_sa_overflow_msb, 0, h_m_denorm_calc );

          // Normal rounding (round to nearest, ties to even)
          const std::uint32_t norm_lsb = ( f_m >> f_h_m_pos_offset ) & 1u;
          const std::uint32_t f_m_rounded = ( f_m + 0x00000fff + norm_lsb );
          const std::uint32_t f_m_rounded_overflow = ( f_m_rounded & f_m_hidden_bit );

          const std::uint32_t m_nan = ( f_m >> f_h_m_pos_offset );
          const std::uint32_t h_em_nan = ( h_e_mask | m_nan );
          const std::uint32_t h_e_norm_overflow_offset = ( f_e_half_bias + 1 );
          const std::uint32_t h_e_norm_overflow = ( h_e_norm_overflow_offset << h_e_pos );
          const std::uint32_t h_e_norm = ( f_e_half_bias << h_e_pos );
          const std::uint32_t h_m_norm = ( f_m_rounded >> f_h_m_pos_offset );
          const std::uint32_t h_em_norm = ( h_e_norm | h_m_norm );

          const std::uint32_t is_h_ndenorm_msb = ( f_h_bias_offset - f_e_amount );
          const std::uint32_t is_f_e_flagged_msb = ( f_h_e_biased_flag - f_e_half_bias );
          const std::uint32_t is_h_denorm_msb = ( ~is_h_ndenorm_msb );
          const std::uint32_t is_f_m_eqz_msb = ( f_m - 1 );
          const std::uint32_t is_h_nan_eqz_msb = ( m_nan - 1 );
          const std::uint32_t is_f_inf_msb = ( is_f_e_flagged_msb & is_f_m_eqz_msb );
          const std::uint32_t is_f_nan_underflow_msb = ( is_f_e_flagged_msb & is_h_nan_eqz_msb );
          const std::uint32_t is_e_overflow_msb = ( h_e_max_value - f_e_half_bias ) & ~is_f_e_flagged_msb;
          const std::uint32_t is_h_inf_msb = ( is_e_overflow_msb | is_f_inf_msb );
          const std::uint32_t is_f_nsnan_msb = ( f_snan - f_snan_mask );
          const std::uint32_t is_m_norm_overflow_msb = ( -f_m_rounded_overflow );
          const std::uint32_t is_f_snan_msb = ( ~is_f_nsnan_msb );

          const std::uint32_t h_em_overflow_result = _uint32_sels( is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm );
          const std::uint32_t h_em_nan_result = _uint32_sels( is_f_e_flagged_msb, h_em_nan, h_em_overflow_result );
          const std::uint32_t h_em_nan_underflow_result = _uint32_sels( is_f_nan_underflow_msb, h_nan_min, h_em_nan_result );
          const std::uint32_t h_em_inf_result = _uint32_sels( is_h_inf_msb, h_e_mask, h_em_nan_underflow_result );
          const std::uint32_t h_em_denorm_result = _uint32_sels( is_h_denorm_msb, h_m_denorm, h_em_inf_result );
          const std::uint32_t h_em_snan_result = _uint32_sels( is_f_snan_msb, h_snan_mask, h_em_denorm_result );
          const std::uint32_t h_result = ( h_s | h_em_snan_result );
          return static_cast<std::uint16_t>(h_result);
      }

      XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t half_to_float( const std::uint16_t h ) noexcept
      {
          const std::uint32_t h_e_mask = 0x00007c00;
          const std::uint32_t h_m_mask = 0x000003ff;
          const std::uint32_t h_s_mask = 0x00008000;
          const std::uint32_t h_f_s_pos_offset = 0x00000010;
          const std::uint32_t h_f_e_pos_offset = 0x0000000d;
          const std::uint32_t h_f_bias_offset = 0x0001c000;
          const std::uint32_t f_e_mask = 0x7f800000;
          const std::uint32_t f_m_mask = 0x007fffff;
          const std::uint32_t h_f_e_denorm_bias = 0x0000007e;
          const std::uint32_t h_f_m_denorm_sa_bias = 0x00000008;
          const std::uint32_t f_e_pos = 0x00000017;
          const std::uint32_t h_e_mask_minus_one = 0x00007bff;
          const std::uint32_t h_e = ( h & h_e_mask );
          const std::uint32_t h_m = ( h & h_m_mask );
          const std::uint32_t h_s = ( h & h_s_mask );
          const std::uint32_t h_e_f_bias = ( h_e + h_f_bias_offset );
          const std::uint32_t h_m_nlz = _uint32_cntlz( h_m );
          const std::uint32_t f_s = ( h_s << h_f_s_pos_offset );
          const std::uint32_t f_e = ( h_e_f_bias << h_f_e_pos_offset );
          const std::uint32_t f_m = ( h_m << h_f_e_pos_offset );
          const std::uint32_t f_em = ( f_e | f_m );
          const std::uint32_t h_f_m_sa = ( h_m_nlz - h_f_m_denorm_sa_bias );
          const std::uint32_t f_e_denorm_unpacked = ( h_f_e_denorm_bias - h_f_m_sa );
          const std::uint32_t h_f_m = ( h_m << h_f_m_sa );
          const std::uint32_t f_m_denorm = ( h_f_m & f_m_mask );
          const std::uint32_t f_e_denorm = ( f_e_denorm_unpacked << f_e_pos );
          const std::uint32_t f_em_denorm = ( f_e_denorm | f_m_denorm );
          const std::uint32_t f_em_nan = ( f_e_mask | f_m );
          const std::uint32_t is_e_eqz_msb = ( h_e - 1 );
          const std::uint32_t is_m_nez_msb = ( -h_m );
          const std::uint32_t is_e_flagged_msb = ( h_e_mask_minus_one - h_e );
          const std::uint32_t is_zero_msb = ( is_e_eqz_msb & ~is_m_nez_msb );
          const std::uint32_t is_inf_msb = ( is_e_flagged_msb & ~is_m_nez_msb );
          const std::uint32_t is_denorm_msb = ( is_m_nez_msb & is_e_eqz_msb );
          const std::uint32_t is_nan_msb = ( is_e_flagged_msb & is_m_nez_msb );
          const std::uint32_t is_zero = ( ( static_cast< std::int32_t>(is_zero_msb) ) >> 31 );
          const std::uint32_t f_zero_result = ( f_em & ~is_zero );
          const std::uint32_t f_denorm_result = _uint32_sels( is_denorm_msb, f_em_denorm, f_zero_result );
          const std::uint32_t f_inf_result = _uint32_sels( is_inf_msb, f_e_mask, f_denorm_result );
          const std::uint32_t f_nan_result = _uint32_sels( is_nan_msb, f_em_nan, f_inf_result );
          const std::uint32_t f_result = ( f_s | f_nan_result );
          return f_result;
      }

      XTD_DEVICE_FUNCTION constexpr inline static std::uint16_t double_to_half( const std::uint64_t d ) noexcept
      {
          static_assert(sizeof(double) == sizeof(uint64_t));

          const std::uint64_t d_s_mask = 0x8000000000000000ULL;
          const std::uint64_t d_e_mask = 0x7ff0000000000000ULL;
          const std::uint64_t d_m_mask = 0x000fffffffffffffULL;
          const std::uint64_t d_m_hidden_bit = 0x0010000000000000ULL;
          const std::uint64_t d_snan_mask = 0x7ff8000000000000ULL;
          const std::uint32_t d_e_pos = 0x00000034;
          const std::uint32_t h_e_pos = 0x0000000a;
          const std::uint32_t h_e_mask = 0x00007c00;
          const std::uint32_t h_snan_mask = 0x00007e00;
          const std::uint32_t h_e_max_value = 0x0000001e;
          const std::uint32_t d_h_s_pos_offset = 0x00000030;
          const std::uint32_t d_h_bias_offset = 0x000003f0;
          const std::uint32_t d_h_m_pos_offset = 0x0000002a;
          const std::uint32_t h_nan_min = 0x00007c01;
          const std::uint32_t d_h_e_biased_flag = 0x0000040e;

          const std::uint64_t d_s = ( d & d_s_mask );
          const std::uint64_t d_e = ( d & d_e_mask );
          const std::uint16_t h_s = static_cast<std::uint16_t>( d_s >> d_h_s_pos_offset );
          const std::uint64_t d_m = ( d & d_m_mask );
          const std::uint32_t d_e_amount = static_cast<std::uint32_t>( d_e >> d_e_pos );
          const std::uint32_t d_e_half_bias = ( d_e_amount - d_h_bias_offset );
          const std::uint64_t d_snan = ( d & d_snan_mask );

          // Denormal shift calculation capped to prevent shift overflow (round to nearest, ties to even)
          const std::uint32_t raw_denorm_sa = ( 1 - d_e_half_bias );
          const std::uint32_t is_sa_overflow_msb = ( 11 - raw_denorm_sa );
          const std::uint32_t denorm_sa_clamped = _uint32_sels( is_sa_overflow_msb, 11, raw_denorm_sa );

          const std::uint64_t d_m_with_hidden = ( d_m | d_m_hidden_bit );
          const std::uint32_t denorm_total_shift = d_h_m_pos_offset + denorm_sa_clamped;
          const std::uint64_t denorm_bias = ( 1ULL << ( denorm_total_shift - 1 ) ) - 1ULL;
          const std::uint32_t denorm_lsb = static_cast<std::uint32_t>( ( d_m_with_hidden >> denorm_total_shift ) & 1ULL );
          const std::uint32_t h_m_denorm_calc = static_cast<std::uint32_t>( ( d_m_with_hidden + denorm_bias + denorm_lsb ) >> denorm_total_shift );
          const std::uint32_t h_m_denorm = _uint32_sels( is_sa_overflow_msb, 0, h_m_denorm_calc );

          // Normal rounding (round to nearest, ties to even)
          const std::uint64_t norm_lsb = ( d_m >> d_h_m_pos_offset ) & 1ULL;
          const std::uint64_t d_m_rounded = ( d_m + 0x000001ffffffffffULL + norm_lsb );
          const std::uint64_t d_m_rounded_overflow = ( d_m_rounded & d_m_hidden_bit );

          const std::uint32_t m_nan = static_cast<std::uint32_t>( d_m >> d_h_m_pos_offset );
          const std::uint32_t h_em_nan = ( h_e_mask | m_nan );
          const std::uint32_t h_e_norm_overflow_offset = ( d_e_half_bias + 1 );
          const std::uint32_t h_e_norm_overflow = ( h_e_norm_overflow_offset << h_e_pos );
          const std::uint32_t h_e_norm = ( d_e_half_bias << h_e_pos );
          const std::uint32_t h_m_norm = static_cast<std::uint32_t>( d_m_rounded >> d_h_m_pos_offset );
          const std::uint32_t h_em_norm = ( h_e_norm | h_m_norm );

          const std::uint32_t is_h_ndenorm_msb = ( d_h_bias_offset - d_e_amount );
          const std::uint32_t is_d_e_flagged_msb = ( d_h_e_biased_flag - d_e_half_bias );
          const std::uint32_t is_h_denorm_msb = ( ~is_h_ndenorm_msb );
          const std::uint32_t is_d_m_eqz_msb = static_cast<std::uint32_t>( ( d_m - 1ULL ) >> 32 );
          const std::uint32_t is_h_nan_eqz_msb = ( m_nan - 1 );
          const std::uint32_t is_d_inf_msb = ( is_d_e_flagged_msb & is_d_m_eqz_msb );
          const std::uint32_t is_d_nan_underflow_msb = ( is_d_e_flagged_msb & is_h_nan_eqz_msb );
          const std::uint32_t is_e_overflow_msb = ( h_e_max_value - d_e_half_bias ) & ~is_d_e_flagged_msb;
          const std::uint32_t is_h_inf_msb = ( is_e_overflow_msb | is_d_inf_msb );
          const std::uint32_t is_d_nsnan_msb = static_cast<std::uint32_t>( ( d_snan - d_snan_mask ) >> 32 );
          const std::uint32_t is_m_norm_overflow_msb = static_cast<std::uint32_t>( -( d_m_rounded_overflow >> 52 ) );
          const std::uint32_t is_d_snan_msb = ( ~is_d_nsnan_msb );

          const std::uint32_t h_em_overflow_result = _uint32_sels( is_m_norm_overflow_msb, h_e_norm_overflow, h_em_norm );
          const std::uint32_t h_em_nan_result = _uint32_sels( is_d_e_flagged_msb, h_em_nan, h_em_overflow_result );
          const std::uint32_t h_em_nan_underflow_result = _uint32_sels( is_d_nan_underflow_msb, h_nan_min, h_em_nan_result );
          const std::uint32_t h_em_inf_result = _uint32_sels( is_h_inf_msb, h_e_mask, h_em_nan_underflow_result );
          const std::uint32_t h_em_denorm_result = _uint32_sels( is_h_denorm_msb, h_m_denorm, h_em_inf_result );
          const std::uint32_t h_em_snan_result = _uint32_sels( is_d_snan_msb, h_snan_mask, h_em_denorm_result );
          const std::uint32_t h_result = ( h_s | h_em_snan_result );
          return static_cast<std::uint16_t>(h_result);
      }

      XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t _uint32_sels( std::uint32_t test, std::uint32_t a, std::uint32_t b ) noexcept
      {
          const std::uint32_t mask = ( ( static_cast< std::int32_t>(test) ) >> 31 );
          const std::uint32_t sel_a = ( a & mask );
          const std::uint32_t sel_b = ( b & ~mask );
          const std::uint32_t result = ( sel_a | sel_b );
          return result;
      }

      XTD_DEVICE_FUNCTION constexpr inline static std::uint32_t _uint32_cntlz( std::uint32_t x ) noexcept
          {
#ifdef __GNUC__
              if ( x == 0 ) { return 0x00000020; }
              std::uint32_t is_x_nez_msb = ( -x );
              std::uint32_t nlz = __builtin_clz( x );
              std::uint32_t result = _uint32_sels( is_x_nez_msb, nlz, 0x00000020 );
              return result;
#else
              if ( x == 0 ) return 0x00000020;
              const std::uint32_t x0 = ( x >> 1 );
              const std::uint32_t x1 = ( x | x0 );
              const std::uint32_t x2 = ( x1 >> 2 );
              const std::uint32_t x3 = ( x1 | x2 );
              const std::uint32_t x4 = ( x3 >> 4 );
              const std::uint32_t x5 = ( x3 | x4 );
              const std::uint32_t x6 = ( x5 >> 8 );
              const std::uint32_t x7 = ( x5 | x6 );
              const std::uint32_t x8 = ( x7 >> 16 );
              const std::uint32_t x9 = ( x7 | x8 );
              const std::uint32_t xA = ( ~x9 );
              const std::uint32_t xB = ( xA >> 1 );
              const std::uint32_t xC = ( xB & 0x55555555 );
              const std::uint32_t xD = ( xA - xC );
              const std::uint32_t xE = ( xD & 0x33333333 );
              const std::uint32_t xF = ( xD >> 2 );
              const std::uint32_t x10 = ( xF & 0x33333333 );
              const std::uint32_t x11 = ( xE + x10 );
              const std::uint32_t x12 = ( x11 >> 4 );
              const std::uint32_t x13 = ( x11 + x12 );
              const std::uint32_t x14 = ( x13 & 0x0f0f0f0f );
              const std::uint32_t x15 = ( x14 >> 8 );
              const std::uint32_t x16 = ( x14 + x15 );
              const std::uint32_t x17 = ( x16 >> 16 );
              const std::uint32_t x18 = ( x16 + x17 );
              const std::uint32_t x19 = ( x18 & 0x0000003f );
              return x19;
#endif // NOT __GNUC__
          }
  };
}
