#include <cublas_v2.h>
#include <iostream>
#include <sstream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <stdexcept>
#include <cassert>

using namespace std;

int main(int argc, char *argv[]) {

  
  bool allow_soft = false;
  if (argc != 2 && argc != 4) {
    throw std::runtime_error("Must provide number of gpus to pick");
  }
  int gpus2pick = 1;
  if (argc == 4) {
    assert(string(argv[1]) == "--allow-soft");
    allow_soft = string(argv[2]) == "true";
    gpus2pick = atoi(argv[3]);
  } else {
    gpus2pick = atoi(argv[1]);
  }

//  cerr << "Going to pick " << gpus2pick << " gpus" << endl;

  // Check that we have a gpu available
  int num_gpus = 0;

  cudaError_t e = cudaGetDeviceCount(&num_gpus);

//  cerr << "Num gpus on this machine: " << num_gpus << endl;

  int device = 0;
  int count_picked = 0;
  std::ostringstream ss;
  while (device < num_gpus && count_picked < gpus2pick) {
    cudaSetDevice(device);
    cudaError_t e = cudaDeviceSynchronize(); // << CUDA context gets created here.
    if (e == cudaSuccess) {
//      cerr << "Device " << device << " available" << endl;
      if (count_picked == 0) {
        ss << device;
      } else {
        ss << "," << device;
      }
      count_picked++;
    }
    device++;
  }
  if (count_picked == gpus2pick || allow_soft) {
    cerr << count_picked << " gpus picked" << endl;
    cout << ss.str();
  } else {
    cerr << "only " << count_picked << " out of " << gpus2pick << " GPUs available" << endl;
    cout << "-1";
  }
}

