#include <iostream>
#include <assert.h>
#include <cudnn.h>
#include <cuda_fp16.h>

#define checkCUDA(expression)                              \
{                                                          \
  cudaError_t status = (expression);                       \
  if (status != cudaSuccess) {                             \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudaGetErrorString(status) << std::endl;  \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define checkCUDNN(expression)                             \
{                                                          \
  cudnnStatus_t status = (expression);                     \
  if (status != CUDNN_STATUS_SUCCESS) {                    \
    std::cerr << "Error on line " << __LINE__ << ": "      \
              << cudnnGetErrorString(status) << std::endl; \
    std::exit(EXIT_FAILURE);                               \
  }                                                        \
}

#define FLOAT_T __half

void init_input(FLOAT_T *ptr, int size, float val) {
  for (int i = 0; i < size; i++) {
    ptr[i]  = __float2half(val);
  }
}

template<typename T>
float FloatLoader(T d) {
  return static_cast<float>(d);
}

template<>
float FloatLoader<__half>(__half d) {
  return __half2float(d);
}

template<typename T>
void print_output(const T* ptr, int size, const char* message,
                  int lines = 10) {
  checkCUDA(cudaDeviceSynchronize());

  const int num_per_line = 20;
  int limit = INT_MAX;
  if (lines != -1) {
    limit = lines * num_per_line;
  }

  printf("%s (showing %d elements):\n", message, std::min(size, limit));
  for (int i = 0; i < std::min(size, limit); i++) {
    printf("%lf, ", FloatLoader<T>(ptr[i]));
    if ((i + 1) % num_per_line == 0) {
      printf("\n");
    }
  }
  printf("\n");
}

int main(int argc, char const *argv[]) {

  int algo_idx = 0;
  if (argc > 1) {
    algo_idx = atoi(argv[1]);
  }

  int N = 1, C = 8, H = 16, W = 16;
  int K = 32, R = 3, S = 3;

  int conv_pads[] = {0, 0};
  int conv_strides[] = {1, 1};
  int conv_dilations[] = {1, 1};

  cudnnDataType_t dataTypeT = CUDNN_DATA_HALF;
  cudnnDataType_t computeTypeT = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensorFormatT = CUDNN_TENSOR_NHWC;
  cudnnConvolutionMode_t convModeT = CUDNN_CROSS_CORRELATION;
  cudnnFusedOps_t fusedOpsT = CUDNN_FUSED_SCALE_BIAS_ACTIVATION_CONV_BNSTATS;
  cudnnFusedOpsPointerPlaceHolder_t ptrPlaceholderT = CUDNN_PTR_16B_ALIGNED;
  cudnnBatchNormMode_t batchNormModeT = CUDNN_BATCHNORM_SPATIAL_PERSISTENT;

  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/input_descriptor,
                                        /*format=*/tensorFormatT,
                                        /*dataType=*/dataTypeT,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));

  cudnnTensorDescriptor_t eq_scale_bias_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&eq_scale_bias_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/eq_scale_bias_descriptor,
                                        /*format=*/tensorFormatT,
                                        /*dataType=*/dataTypeT,
                                        /*batch_size=*/1,
                                        /*channels=*/C,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(cudnnSetActivationDescriptor(
      /*activationDesc=*/activation_descriptor,
      /*mode=*/CUDNN_ACTIVATION_RELU,
      /*reluNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
      /*coef=*/0.000000));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(cudnnSetConvolution2dDescriptor(
      /*convDesc=*/convolution_descriptor,
      /*pad_height=*/conv_pads[0],
      /*pad_width=*/conv_pads[1],
      /*vertical_stride=*/conv_strides[0],
      /*horizontal_stride=*/conv_strides[1],
      /*dilation_height=*/conv_dilations[0],
      /*dilation_width=*/conv_dilations[1],
      /*mode=*/convModeT,
      /*computeType=*/computeTypeT));
  cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH);

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(/*filterDesc=*/kernel_descriptor,
                                        /*dataType=*/dataTypeT,
                                        /*format=*/tensorFormatT,
                                        /*out_channels=*/K,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/R,
                                        /*kernel_width=*/S));

  int N_, K_, P, Q;
  checkCUDNN(cudnnGetConvolution2dForwardOutputDim(
      /*convDesc=*/convolution_descriptor,
      /*inputTensorDesc=*/input_descriptor,
      /*filterDesc=*/kernel_descriptor,
      /*out_batch_size=*/&N_,
      /*out_channels=*/&K_,
      /*out_image_height=*/&P,
      /*out_image_width=*/&Q));
  assert(N_ == N);
  assert(K_ == K);
                                                  
  cudnnTensorDescriptor_t output_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/output_descriptor,
                                        /*format=*/tensorFormatT,
                                        /*dataType=*/dataTypeT,
                                        /*batch_size=*/N,
                                        /*channels=*/K,
                                        /*image_height=*/P,
                                        /*image_width=*/Q));

  cudnnTensorDescriptor_t output_stats_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&output_stats_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/output_stats_descriptor,
                                        /*format=*/tensorFormatT,
                                        /*dataType=*/computeTypeT,
                                        /*batch_size=*/1,
                                        /*channels=*/K,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  printf("LOG >>> Input  dims: (%d, %d, %d, %d)\n", N, C, H, W);
  printf("LOG >>> Filter dims: (%d, %d, %d, %d)\n", K, C, R, S);
  printf("LOG >>> Output dims: (%d, %d, %d, %d)\n", N, K, P, Q);

  cudnnFusedOpsConstParamPack_t const_param;
  checkCUDNN(cudnnCreateFusedOpsConstParamPack(&const_param, fusedOpsT));
  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_XDESC,
      input_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_XDATA_PLACEHOLDER, 
      &ptrPlaceholderT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_BN_MODE, 
      &batchNormModeT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_BN_EQSCALEBIAS_DESC,
      eq_scale_bias_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_BN_EQSCALE_PLACEHOLDER, 
      &ptrPlaceholderT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_BN_EQBIAS_PLACEHOLDER, 
      &ptrPlaceholderT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_ACTIVATION_DESC,
      activation_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_CONV_DESC,
      convolution_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_WDESC,
      kernel_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_WDATA_PLACEHOLDER, 
      &ptrPlaceholderT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_YDESC,
      output_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_YDATA_PLACEHOLDER, 
      &ptrPlaceholderT)); 

	checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param, 
      CUDNN_PARAM_YSTATS_DESC,
      output_stats_descriptor));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_YSUM_PLACEHOLDER, 
      &ptrPlaceholderT));

  checkCUDNN(cudnnSetFusedOpsConstParamPackAttribute(
      const_param,
      CUDNN_PARAM_YSQSUM_PLACEHOLDER, 
      &ptrPlaceholderT));

  cudnnFusedOpsPlan_t plan;
  checkCUDNN(cudnnCreateFusedOpsPlan(&plan, fusedOpsT));
	size_t workspace_bytes = 0;
  checkCUDNN(cudnnMakeFusedOpsPlan(
      /*handle=*/cudnn, 
      /*plan=*/plan, 
      /*constPack=*/const_param, 
      /*workspaceSizeInBytes=*/&workspace_bytes));
            
  printf("LOG >>> Workspace size (bytes): %ld\n", workspace_bytes);

  void* d_workspace{nullptr};
  if (workspace_bytes != 0) {
    checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));
  }
  
  int x_size = N * C * H * W;
  int y_size = N * K * Q * P;
  int w_size = C * K * R * S;
  int eq_g_size = 1 * C * 1 * 1;
  int eq_b_size = 1 * C * 1 * 1;
  int y_m_size = 1 * K * 1 * 1;
  int y_s_size = 1 * K * 1 * 1;

  int x_bytes = x_size * sizeof(FLOAT_T);
  int y_bytes = y_size * sizeof(FLOAT_T);
  int w_bytes = w_size * sizeof(FLOAT_T);
  int eq_g_bytes = eq_g_size * sizeof(FLOAT_T);
  int eq_b_bytes = eq_b_size * sizeof(FLOAT_T);
  int y_m_bytes = y_m_size * sizeof(float);
  int y_s_bytes = y_s_size * sizeof(float);

  FLOAT_T *x;
  FLOAT_T *y;
  FLOAT_T *w;
  FLOAT_T *eq_g;
  FLOAT_T *eq_b;
  float *y_m;
  float *y_s;
  checkCUDA(cudaMallocManaged(&x, x_bytes));
  checkCUDA(cudaMallocManaged(&y, y_bytes));
  checkCUDA(cudaMallocManaged(&w, w_bytes));
  checkCUDA(cudaMallocManaged(&eq_g, eq_g_bytes));
  checkCUDA(cudaMallocManaged(&eq_b, eq_b_bytes));
  checkCUDA(cudaMallocManaged(&y_m, y_m_bytes));
  checkCUDA(cudaMallocManaged(&y_s, y_s_bytes));

  init_input(x, x_size, 1.0);
  init_input(w, w_size, 1.0);
  init_input(eq_g, eq_g_size, 1.0);
  init_input(eq_b, eq_b_size, 0.0);

  cudnnFusedOpsVariantParamPack_t variant_param;
  checkCUDNN(cudnnCreateFusedOpsVariantParamPack(&variant_param, fusedOpsT));

  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_XDATA, 
                                                       (void*)x));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_BN_EQSCALE, 
                                                       (void*)eq_g));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_BN_EQBIAS, 
                                                       (void*)eq_b));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_WDATA, 
                                                       (void*)w));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_YDATA, 
                                                       (void*)y));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_YSUM, 
                                                       (void*)y_m));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param, 
                                                       CUDNN_PTR_YSQSUM, 
                                                       (void*)y_s));

  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(variant_param,
                                                       CUDNN_PTR_WORKSPACE, 
                                                       (void*)d_workspace));
  checkCUDNN(cudnnSetFusedOpsVariantParamPackAttribute(
      variant_param,
      CUDNN_SCALAR_SIZE_T_WORKSPACE_SIZE_IN_BYTES, 
      &workspace_bytes));

  checkCUDNN(cudnnFusedOpsExecute(cudnn, 
                                  plan, 
                                  variant_param));

  checkCUDA(cudaDeviceSynchronize());
  assert(y[0] == 9.0 * C);
  assert(y_m[0] == 9.0 * C * P * Q);
  assert(y_s[0] == 9.0 * C * 9.0 * C * P * Q);
  printf("LOG >>> Results look good!\n");

  // print_output(y, y_size, "Y out:", 3);
  // print_output(y_m, y_m_size, "YSUM out:", 3);
  // print_output(y_s, y_s_size, "YSQSUM out:", 3);

  checkCUDNN(cudnnDestroyTensorDescriptor(input_descriptor));
  checkCUDNN(cudnnDestroyTensorDescriptor(output_descriptor));
  checkCUDNN(cudnnDestroyTensorDescriptor(eq_scale_bias_descriptor));
  checkCUDNN(cudnnDestroyTensorDescriptor(output_stats_descriptor));
  checkCUDNN(cudnnDestroyFilterDescriptor(kernel_descriptor));
  checkCUDNN(cudnnDestroyActivationDescriptor(activation_descriptor));
  checkCUDNN(cudnnDestroyConvolutionDescriptor(convolution_descriptor));
  checkCUDNN(cudnnDestroyFusedOpsConstParamPack(const_param));
  checkCUDNN(cudnnDestroyFusedOpsPlan(plan));
  checkCUDNN(cudnnDestroyFusedOpsVariantParamPack(variant_param));
  checkCUDNN(cudnnDestroy(cudnn));

  checkCUDA(cudaFree(x));
  checkCUDA(cudaFree(w));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(eq_g));
  checkCUDA(cudaFree(eq_b));
  checkCUDA(cudaFree(y_m));
  checkCUDA(cudaFree(y_s));
  checkCUDA(cudaFree(d_workspace));

}
