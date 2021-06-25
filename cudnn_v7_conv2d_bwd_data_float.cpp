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

#define FLOAT_T float

void init_input(FLOAT_T *ptr, int size) {
  for (int i = 0; i < size; i++) {
    ptr[i]  = static_cast<FLOAT_T>(rand()) / RAND_MAX;
  }
}

void print_output(const FLOAT_T* ptr, int size, const char* message,
                  int lines = 10) {
  checkCUDA(cudaDeviceSynchronize());

  const int num_per_line = 20;
  int limit = INT_MAX;
  if (lines != -1) {
    limit = lines * num_per_line;
  }

  printf("%s (showing %d elements):\n", message, std::min(size, limit));
  for (int i = 0; i < std::min(size, limit); i++) {
    printf("%lf, ", ptr[i]);
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

  cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
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
  
  printf("LOG >>> InputGrad  dims: (%d, %d, %d, %d)\n", N, C, H, W);
  printf("LOG >>> Filter     dims: (%d, %d, %d, %d)\n", K, C, R, S);
  printf("LOG >>> OutputGrad dims: (%d, %d, %d, %d)\n", N, K, P, Q);

  cudnnConvolutionBwdDataAlgo_t convolution_algorithm =
  	  static_cast<cudnnConvolutionBwdDataAlgo_t>(algo_idx);
  printf("LOG >>> Selecting Algorithm (%d)\n", algo_idx);
  
  size_t workspace_bytes = 0;
  checkCUDNN(cudnnGetConvolutionBackwardDataWorkspaceSize(
                 /*handle=*/cudnn,
                 /*wDesc=*/kernel_descriptor,
                 /*dyDesc=*/output_descriptor,
                 /*convDesc=*/convolution_descriptor,
                 /*dxDesc=*/input_descriptor,
                 /*algo=*/convolution_algorithm,
                 /*sizeInBytes=*/&workspace_bytes));
  printf("LOG >>> Workspace size (bytes): %ld\n", workspace_bytes);

  void* d_workspace{nullptr};
  if (workspace_bytes != 0) {
    checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));
  }
  
  int x_size = N * C * H * W;
  int y_size = N * K * Q * P;
  int w_size = C * K * R * S;

  int x_bytes = x_size * sizeof(FLOAT_T);
  int y_bytes = y_size * sizeof(FLOAT_T);
  int w_bytes = w_size * sizeof(FLOAT_T);

  FLOAT_T *x;
  FLOAT_T *y;
  FLOAT_T *w;
  checkCUDA(cudaMallocManaged(&x, x_bytes));
  checkCUDA(cudaMallocManaged(&y, y_bytes));
  checkCUDA(cudaMallocManaged(&w, w_bytes));

  srand(3);
  init_input(x, x_size);
  init_input(y, y_size);
  init_input(w, w_size);

  const FLOAT_T alpha = 1.0;
  const FLOAT_T beta = 0.0;
  checkCUDNN(cudnnConvolutionBackwardData(
                 /*handle=*/cudnn,
                 /*alpha=*/&alpha,
                 /*wDesc=*/kernel_descriptor,
                 /*w=*/w,
                 /*dyDesc=*/output_descriptor,
                 /*dy=*/y,
                 /*convDesc=*/convolution_descriptor,
                 /*algo=*/convolution_algorithm,
                 /*workSpace=*/d_workspace,
                 /*workSpaceSizeInBytes=*/workspace_bytes,
                 /*beta=*/&beta,
                 /*dxDesc=*/input_descriptor,
                 /*dx=*/x));

  print_output(x, x_size, "dx out:", -1);

  checkCUDA(cudaFree(w));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(x));
  if (workspace_bytes != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
}
