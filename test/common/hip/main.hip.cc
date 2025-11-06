#include <cstdlib>
#include <iostream>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

// HIP headers
#include <hip/hip_runtime.h>

int main(int argc, char* argv[]) {
  // check that the system has one or more AMD GPUs
  int deviceCount;
  hipError_t status = hipGetDeviceCount(&deviceCount);
  if (status != hipSuccess || deviceCount == 0) {
    std::cout << "No AMD GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

  // run the Catch tests
  return Catch::Session().run(argc, argv);
}
