//
// Created by JEONGHYUNLEE on 2020/10/12.
//

#ifndef CAFFE_FUNCTIONS_COMMON_H
#define CAFFE_FUNCTIONS_COMMON_H

#include <stdio.h>
#include <string>
#include <vector>

typedef float Dtype;

void print_tensor(std::string name, const Dtype* tensor, const std::vector<int> shape) {
  printf("%s\n",name.c_str());
  for (int f=0;f<shape[0];f++) {
    for (int c=0;c<shape[1];c++) {
      for (int h=0;h<shape[2];h++) {
        for (int w=0;w<shape[3];w++) {
          const int offset = ((f * shape[1] + c) * shape[2] + h) * shape[3] + w;
          printf("%f ",tensor[offset]);
        }
      }
      printf(" | ");
    }
    printf("\n");
  }
  return;
}

void set_values(Dtype* tensor, std::vector<Dtype> values, const int size) {
  for (int i=0;i<size;i++){
    tensor[i] = values[i];
  }
  return;
}

#endif //CAFFE_FUNCTIONS_COMMON_H

