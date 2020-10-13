# Caffe API Archieve

Some APIs for the sparse learning and the quantization of DNN on Caffe framework.

The sparse learning is based on filter pruning:
https://arxiv.org/pdf/1608.08710.pdf

```math
\min_{w} L(w)+\lambda \|w_F\|_0
```

The quantization is based on k-means and gradient based fine-tuning:
https://arxiv.org/abs/1510.00149

