#CUDNN_FRONTEND_DIR=./cudnn-frontend/include/
CUDNN_FRONTEND_DIR=/home/cudnn_frontend/include/
CXXFLAGS=-DNV_CUDNN_DISABLE_EXCEPTION -lcudnn

cudnn_v8_matmul_pointwise_fp16.out: cudnn_v8_matmul_pointwise_fp16.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_fused_ops.out: cudnn_v7_fused_ops.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_conv2d_fwd_float.out: cudnn_v8_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_conv2d_bwd_filter_float.out: cudnn_v8_conv2d_bwd_filter_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_conv2d_bwd_data_float.out: cudnn_v8_conv2d_bwd_data_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_conv2d_fwd_int8x4.out: cudnn_v8_conv2d_fwd_int8x4.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_conv2d_fwd_int8x32.out: cudnn_v8_conv2d_fwd_int8x32.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_fused_conv2d_fwd_float.out: cudnn_v8_fused_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v8_fused_conv2d_fwd_int8x4.out: cudnn_v8_fused_conv2d_fwd_int8x4.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_fwd_float.out: cudnn_v7_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_bwd_filter_float.out: cudnn_v7_conv2d_bwd_filter_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_bwd_data_float.out: cudnn_v7_conv2d_bwd_data_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_fwd_int8x4.out: cudnn_v7_conv2d_fwd_int8x4.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_conv2d_fwd_int8x32.out: cudnn_v7_conv2d_fwd_int8x32.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_fused_conv2d_fwd_float.out: cudnn_v7_fused_conv2d_fwd_float.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

cudnn_v7_fused_conv2d_fwd_int8x4.out: cudnn_v7_fused_conv2d_fwd_int8x4.cpp
	nvcc $< -o $@ -I ${CUDNN_FRONTEND_DIR} ${CXXFLAGS}

clean:
	rm -rf *.out

