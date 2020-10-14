//
// Created by JEONGHYUNLEE on 2020/09/28.
//

#include <stdio.h>
#include <caffe/caffe.hpp>
#include "./common.hpp"

using namespace caffe;
typedef float Dtype;


// Wrapper
Dtype caffe_gpu_filter_norm(const std::vector<int> shape,
                            const Dtype *target,
                            const int filter_idx) {
  const int filter_size = shape[1] * shape[2] * shape[3];
  Dtype result;
  CUBLAS_CHECK(cublasSnrm2(Caffe::cublas_handle(),
                           filter_size,
                           target + filter_idx * filter_size,
                           1,
                           &result));
  return result;
}


// Test Code
int main() {
  vector<int> weight_shape = {2, 2, 2, 2};
  shared_ptr <Blob<Dtype>> blob_a(new Blob<Dtype>(weight_shape));

  set_values(blob_a->mutable_cpu_data(), {0, 0, 0, 0,
                                          1, 1, 1, 1,
                                          2, 2, 2, 2,
                                          3, 3, 3, 3}, blob_a->count());

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());

  Dtype result = 0.0;

  //it should be smaller than the number of filters (2)
  const int filter_index = 1;

  result = caffe_gpu_filter_norm(blob_a->shape(), blob_a->gpu_data(), filter_index);

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());

  printf("%dth filter norm: %f", filter_index, result);
  printf("\n");

  return 0;
}
