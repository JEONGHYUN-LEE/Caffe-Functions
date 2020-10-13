//
// Created by JEONGHYUNLEE on 2020/09/28.
//

#include <stdio.h>
#include <caffe/caffe.hpp>
#include "./common.hpp"

using namespace caffe;
typedef float Dtype;

// Kernel Function
__global__ void find_kernel(const int n,
                            const Dtype* target,
                            Dtype* idx,
                            const Dtype value
){
  CUDA_KERNEL_LOOP(index, n) {
    if (target[index] == value) {
      idx[index] = 1.0;
    }
    else {
      idx[index] = 0.0;
    }
  }
}

// Wrapper
void caffe_gpu_find(const int N,
                    const Dtype* target,
                    Dtype* idx,
                    const Dtype value) {
  find_kernel<<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS>>>(N, target, idx, value);
}

// Test Code
int main() {
  vector<int> weight_shape = {2,2,2,2};
  shared_ptr<Blob<Dtype> > blob_a(new Blob<Dtype>(weight_shape));
  shared_ptr<Blob<Dtype> > blob_b(new Blob<Dtype>(weight_shape));

  set_values(blob_a->mutable_cpu_data(), {0,0,0,0,
                                          1,1,1,1,
                                          2,2,2,2,
                                          3,3,3,3}, blob_a->count());

  set_values(blob_b->mutable_cpu_data(), {0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0,
                                          0,0,0,0}, blob_b->count());

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());
  print_tensor("b", blob_b->cpu_data(), blob_b->shape());

  CHECK_EQ(blob_a->count(), blob_b->count());

  caffe_gpu_find(blob_a->count(), blob_a->gpu_data(), blob_b->mutable_gpu_data(), 2);

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());
  print_tensor("b", blob_b->cpu_data(), blob_b->shape());

  return 0;
}
