//
// Created by JEONGHYUNLEE on 2020/09/27.
//

#include "caffe/caffe.hpp"
#include "common.hpp"
#include <stdio.h>

using namespace caffe;
typedef float Dtype;

// Kernel Function
__global__ void replace_kernel(const int n,
                               Dtype *target,
                               const Dtype *idx_mask,
                               const Dtype value) {
  CUDA_KERNEL_LOOP(index, n)
  {
    if (idx_mask[index] == 1.0) {
      target[index] = value;
    }
  }
}

// Wrapper
void caffe_gpu_replace(const int N,
                       Dtype *target,
                       const Dtype *idx_mask,
                       const Dtype value) {
  replace_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, target, idx_mask, value);
}

// Test Code
int main() {
  vector<int> weight_shape = {2, 2, 2, 2};
  shared_ptr <Blob<Dtype>> blob_a(new Blob<Dtype>(weight_shape));
  shared_ptr <Blob<Dtype>> blob_b(new Blob<Dtype>(weight_shape));

  set_values(blob_a->mutable_cpu_data(), {0, 0, 0, 0,
                                          1, 1, 1, 1,
                                          2, 2, 2, 2,
                                          3, 3, 3, 3}, blob_a->count());

  set_values(blob_b->mutable_cpu_data(), {1, 1, 0, 0,
                                          0, 1, 0, 1,
                                          1, 0, 1, 0,
                                          0, 0, 1, 1}, blob_b->count());

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());
  print_tensor("b", blob_b->cpu_data(), blob_b->shape());

  CHECK_EQ(blob_a->count(), blob_b->count());
  caffe_gpu_replace(blob_a->count(), blob_a->mutable_gpu_data(), blob_b->mutable_gpu_data(), 100.0);

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());
  print_tensor("b", blob_b->cpu_data(), blob_b->shape());

  return 0;
}

