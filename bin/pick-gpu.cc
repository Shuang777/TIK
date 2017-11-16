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
  if (argc != 3 && argc != 5) {
    throw std::runtime_error("Must provide number of gpus to pick");
  }
  int gpus2pick = 1;
  int gpu_id = -1;
  if (argc == 5) {
    assert(string(argv[1]) == "--allow-soft");
    allow_soft = string(argv[2]) == "true";
    gpus2pick = atoi(argv[3]);
    gpu_id = atoi(argv[4]);
  } else {
    gpus2pick = atoi(argv[1]);
    gpu_id = atoi(argv[2]);
  }

  if (gpu_id != -1 && gpus2pick != 1) 
    throw std::runtime_error("Cannot pick more than two gpus with gpu_id != -1");

//  cerr << "Going to pick " << gpus2pick << " gpus" << endl;

  // Check that we have a gpu available
  int num_gpus = 0;

  cudaError_t e = cudaGetDeviceCount(&num_gpus);

//  cerr << "Num gpus on this machine: " << num_gpus << endl;

  int device = 0;
  int count_picked = 0;
  int count_checked = 1;
  std::ostringstream ss;
  while (device < num_gpus && count_picked < gpus2pick) {
    cudaSetDevice(device);
    cudaError_t e = cudaDeviceSynchronize(); // << CUDA context gets created here.
    if (e == cudaSuccess) {
//      cerr << "Device " << device << " available" << endl;
      if (gpu_id == -1) {
        if (count_picked == 0) {
          ss << device;
        } else {
          ss << "," << device;
        }
        count_picked++;
      } else if (gpu_id == count_checked){
        ss << device;
        count_picked++;
      } else {
        count_checked++;
      }
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

