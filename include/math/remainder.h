
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T remainder(T x, T y) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::remainder(x, y);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::remainder(x, y);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::remainder(x, y);
#else
  // standard C++ code
  return std::remainder(x, y);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float remainderf(float x, float y) {
  return remainder(x, y);
}

XTD_DEVICE_FUNCTION inline constexpr long double remainderl(long double x, long double y) {
  return remainder(x, y);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double remainder(T x, T y) {
  return remainder(static_cast<double>(x), static_cast<double>(y));
}

} // namespace xtd
