CC = nvcc

CXXFLAGS = -std=c++14 -O2 -DUSE_OPENCV

CAFFE_ROOT = /local_vol/ETRI2020/caffe

INCLUDE = -I $(CAFFE_ROOT)/build/install/include \

LIBRARY = -L $(CAFFE_ROOT)/build/lib \
	  -lboost_system \
	  -lglog \
	  -lcaffe \
	  -lcublas \
	  -lprotobuf

all:
	$(CC) $(INCLUDE) $(OBJS) $(CXXFLAGS) $(PROJECT).cu -o $(PROJECT) $(LIBRARY)
