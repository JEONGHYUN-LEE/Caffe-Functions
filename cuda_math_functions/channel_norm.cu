//
// Created by JEONGHYUNLEE on 2020/09/28.
//

#include <stdio.h>
#include <caffe/caffe.hpp>
#include "./common.hpp"

using namespace caffe;
typedef float Dtype;


// Wrapper
Dtype caffe_gpu_channel_norm(const std::vector<int> shape,
                             const Dtype *target,
                             const int channel_idx) {
  const int filter_number = shape[0];
  const int filter_size = shape[1] * shape[2] * shape[3];
  const int channel_size = shape[2] * shape[3];
  Dtype filter_result;
  Dtype result = 0;
  for (int filter_idx = 0; filter_idx < filter_number; filter_idx++) {
    CUBLAS_CHECK(cublasSnrm2(Caffe::cublas_handle(),
                             channel_size,
                             target + filter_idx * filter_size+channel_idx*channel_size,
                             1,
                             &filter_result));
    result+=filter_result*filter_result;
  }
  result = sqrt(result);
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

  //it should be smaller than the number of channels (2)
  const int filter_index = 1;

  result = caffe_gpu_channel_norm(blob_a->shape(), blob_a->gpu_data(), filter_index);

  print_tensor("a", blob_a->cpu_data(), blob_a->shape());

  printf("%dth channel norm: %f", filter_index, result);
  printf("\n");

  return 0;
}
