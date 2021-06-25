#include <iostream>
#include <assert.h>
#include <cudnn.h>

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

#define INT_T int8_t

void init_input(INT_T *ptr, int size) {
  for (int i = 0; i < size; i++) {
    ptr[i]  = static_cast<INT_T>(rand() % 2);
  }
}

void print_output(const INT_T* ptr, int size, const char* message,
                  int lines = 10) {
  checkCUDA(cudaDeviceSynchronize());

  const int num_per_line = 20;
  int limit = INT_MAX;
  if (lines != -1) {
    limit = lines * num_per_line;
  }

  printf("%s (showing %d elements):\n", message, std::min(size, limit));
  for (int i = 0; i < std::min(size, limit); i++) {
    printf("%d, ", static_cast<int>(ptr[i]));
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

  int N = 1, C = 4, H = 16, W = 16;
  int K = 12, R = 3, S = 3;

  int conv_pads[] = {0, 0};
  int conv_strides[] = {1, 1};
  int conv_dilations[] = {1, 1};

  cudnnDataType_t dataType = CUDNN_DATA_INT8x4;
  cudnnDataType_t computeType = CUDNN_DATA_INT32;
  cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW_VECT_C;
  cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;

  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));

  cudnnTensorDescriptor_t input_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/input_descriptor,
                                        /*format=*/tensorFormat,
                                        /*dataType=*/dataType,
                                        /*batch_size=*/N,
                                        /*channels=*/C,
                                        /*image_height=*/H,
                                        /*image_width=*/W));

  cudnnTensorDescriptor_t bias_descriptor;
  checkCUDNN(cudnnCreateTensorDescriptor(&bias_descriptor));
  checkCUDNN(cudnnSetTensor4dDescriptor(/*tensorDesc=*/bias_descriptor,
                                        /*format=*/tensorFormat,
                                        /*dataType=*/dataType,
                                        /*batch_size=*/N,
                                        /*channels=*/K,
                                        /*image_height=*/1,
                                        /*image_width=*/1));

  cudnnFilterDescriptor_t kernel_descriptor;
  checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
  checkCUDNN(cudnnSetFilter4dDescriptor(/*filterDesc=*/kernel_descriptor,
                                        /*dataType=*/dataType,
                                        /*format=*/tensorFormat,
                                        /*out_channels=*/K,
                                        /*in_channels=*/C,
                                        /*kernel_height=*/R,
                                        /*kernel_width=*/S));

  cudnnConvolutionDescriptor_t convolution_descriptor;
  checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
  checkCUDNN(
      cudnnSetConvolution2dDescriptor(/*convDesc=*/convolution_descriptor,
                                      /*pad_height=*/conv_pads[0],
                                      /*pad_width=*/conv_pads[1],
                                      /*vertical_stride=*/conv_strides[0],
                                      /*horizontal_stride=*/conv_strides[1],
                                      /*dilation_height=*/conv_dilations[0],
                                      /*dilation_width=*/conv_dilations[1],
                                      /*mode=*/convMode,
                                      /*computeType=*/computeType));
  cudnnSetConvolutionMathType(convolution_descriptor, CUDNN_TENSOR_OP_MATH);

  cudnnActivationDescriptor_t activation_descriptor;
  checkCUDNN(cudnnCreateActivationDescriptor(&activation_descriptor));
  checkCUDNN(
      cudnnSetActivationDescriptor(/*activationDesc=*/activation_descriptor,
                                   /*mode=*/CUDNN_ACTIVATION_RELU,
                                   /*reluNanOpt=*/CUDNN_NOT_PROPAGATE_NAN,
                                   /*coef=*/0.000000));

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
                                        /*format=*/tensorFormat,
                                        /*dataType=*/dataType,
                                        /*batch_size=*/N,
                                        /*channels=*/K,
                                        /*image_height=*/P,
                                        /*image_width=*/Q));
  
  printf("LOG >>> Input  dims: (%d, %d, %d, %d)\n", N, C, H, W);
  printf("LOG >>> Filter dims: (%d, %d, %d, %d)\n", K, C, R, S);
  printf("LOG >>> Bias   dims: (%d, %d, %d, %d)\n", 1, K, 1, 1);
  printf("LOG >>> Output dims: (%d, %d, %d, %d)\n", N, K, P, Q);

  cudnnConvolutionFwdAlgo_t convolution_algorithm =
  	  static_cast<cudnnConvolutionFwdAlgo_t>(algo_idx);
  printf("LOG >>> Selecting Algorithm (%d)\n", algo_idx);
  
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
                 /*handle=*/cudnn,
                 /*xDesc=*/input_descriptor,
                 /*wDesc=*/kernel_descriptor,
                 /*convDesc=*/convolution_descriptor,
                 /*yDesc=*/output_descriptor,
                 /*algo=*/convolution_algorithm,
                 /*sizeInBytes=*/&workspace_bytes));
  printf("LOG >>> Workspace size (bytes): %ld\n", workspace_bytes);

  void* d_workspace{nullptr};
  if (workspace_bytes != 0) {
    checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));
  }
  
  int x_size = N * C * H * W;
  int y_size = N * K * Q * P;
  int b_size = 1 * K * 1 * 1;
  int w_size = C * K * R * S;

  int x_bytes = x_size * sizeof(INT_T);
  int y_bytes = y_size * sizeof(INT_T);
  int b_bytes = b_size * sizeof(INT_T);
  int w_bytes = w_size * sizeof(INT_T);

  INT_T *x;
  INT_T *y;
  INT_T *b;
  INT_T *w;
  checkCUDA(cudaMallocManaged(&x, x_bytes));
  checkCUDA(cudaMallocManaged(&y, y_bytes));
  checkCUDA(cudaMallocManaged(&b, b_bytes));
  checkCUDA(cudaMallocManaged(&w, w_bytes));

  srand(3);
  init_input(x, x_size);
  init_input(y, y_size);
  init_input(b, b_size);
  init_input(w, w_size);

  const float alpha = 1.0;
  const float beta = 0.0;
  checkCUDNN(cudnnConvolutionBiasActivationForward(
                 /*handle=*/cudnn,
                 /*alpha=*/&alpha,
                 /*xDesc=*/input_descriptor,
                 /*x=*/x,
                 /*wDesc=*/kernel_descriptor,
                 /*w=*/w,
                 /*convDesc=*/convolution_descriptor,
                 /*algo=*/convolution_algorithm,
                 /*workSpace=*/d_workspace,
                 /*workSpaceSizeInBytes=*/workspace_bytes,
                 /*beta=*/&beta,
                 /*zDesc=*/output_descriptor,
                 /*z=*/y,
                 /*biasDesc=*/bias_descriptor,
                 /*bias=*/b,
                 /*activationDesc=*/activation_descriptor,
                 /*yDesc=*/output_descriptor,
                 /*y=*/y));

  print_output(y, y_size, "Y out:", -1);

  checkCUDA(cudaFree(w));
  checkCUDA(cudaFree(b));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(x));
  if (workspace_bytes != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
}
