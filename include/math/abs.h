
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T abs(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::abs(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::abs(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::abs(x);
#else
  // standard C++ code
  return std::abs(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float fabsf(float x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::fabsf(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::fabsf(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::fabsf(x);
#else
  // standard C++ code
  return fabsf(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr long double fabsl(long double x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::fabsl(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::fabsl(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::fabsl(x);
#else
  // standard C++ code
  return fabsl(x);
#endif
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double fabs(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::fabs(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::fabs(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::fabs(x);
#else
  // standard C++ code
  return fabs(x);
#endif
}

} // namespace xtd
