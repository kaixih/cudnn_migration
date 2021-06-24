CUDNN_FRONTEND_DIR=./cudnn-frontend/include/
CXXFLAGS=-DNV_CUDNN_DISABLE_EXCEPTION -lcudnn

cudnn_v8_conv2d_fwd_float.out: cudnn_v8_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_fused_conv2d_fwd_float.out: cudnn_v8_fused_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_fwd_float.out: cudnn_v7_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_fused_conv2d_fwd_float.out: cudnn_v7_fused_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

