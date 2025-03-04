
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) &&                   \
    !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T sqrt(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::sqrt(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::sqrt(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::sqrt(x);
#else
  // standard C++ code
  return std::sqrt(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float sqrtf(float x) { return sqrt(x); }

XTD_DEVICE_FUNCTION inline constexpr long double sqrtl(long double x) {
  return sqrt(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double sqrt(T x) {
  return sqrt(static_cast<double>(x));
}

} // namespace xtd
