#include <cstdlib>
#include <iostream>

// Catch2 headers
#define CATCH_CONFIG_NO_POSIX_SIGNALS
#define CATCH_CONFIG_RUNNER
#include <catch.hpp>

// CUDA headers
#include <cuda_runtime.h>

int main(int argc, char* argv[]) {
  // check that the system has one or more NVIDIA GPUs
  int deviceCount;
  cudaError_t status = cudaGetDeviceCount(&deviceCount);
  if (status != cudaSuccess || deviceCount == 0) {
    std::cout << "No NVIDIA GPUs found, the test will be skipped.\n\n";
    exit(EXIT_SUCCESS);
  }

  // run the Catch tests
  return Catch::Session().run(argc, argv);
}
