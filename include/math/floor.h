
#pragma once

#include "internal/defines.h"
#include <type_traits>

#if !defined(XTD_TARGET_CUDA) && !defined(XTD_TARGET_HIP) && !defined(XTD_TARGET_SYCL)
#include <cmath>
#endif

namespace xtd {

template <typename T, typename = std::enable_if_t<std::is_floating_point_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr T floor(T x) {
#if defined(XTD_TARGET_CUDA)
  // CUDA device code
  return ::floor(x);
#elif defined(XTD_TARGET_HIP)
  // HIP/ROCm device code
  return ::floor(x);
#elif defined(XTD_TARGET_SYCL)
  // SYCL device code
  return sycl::floor(x);
#else
  // standard C++ code
  return std::floor(x);
#endif
}

XTD_DEVICE_FUNCTION inline constexpr float floorf(float x) {
  return floor(x);
}

XTD_DEVICE_FUNCTION inline constexpr long double floorl(long double x) {
  return floor(x);
}

template <typename T, typename = std::enable_if_t<std::is_integral_v<T>>>
XTD_DEVICE_FUNCTION inline constexpr double floor(T x) {
	return floor(static_cast<double>(x));
}

} // namespace xtd
