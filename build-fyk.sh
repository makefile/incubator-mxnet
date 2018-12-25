#make clean_all
make -j8 \
    USE_OPENCV=0 \
    USE_CUDA=1 USE_CUDA_PATH=/usr/local/cuda-9.2 USE_CUDNN=1 \
    USE_SIGNAL_HANDLER=1 \
    USE_DIST_KVSTORE=1 \
    USE_NCCL=1 \
#ADD_LDFLAGS=~/local_install/lib
# build with distribute train need downloading: USE_DIST_KVSTORE=1

if [[ $? -eq 0 ]]; then
    cd python; python setup-fyk.py install; cd ..
fi

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# DO not use openblas, maybe cause speed problem!! use atlas instead
# modify the 3rd-part/Makefile, and compile in place where has network connection
# see https://discuss.gluon.ai/t/topic/8884/5
# https://github.com/apache/incubator-mxnet/issues/8671
