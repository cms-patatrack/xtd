
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T ceil(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::ceil(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::ceil(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::ceil(x);
#else
  // standard C++ code
  return std::ceil(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float ceilf(float x) {
  return ceil(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double ceill(long double x) {
  return ceil(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double ceil(T x) {
	return ceil(static_cast<double>(x));
}

} // namespace xtd
