#include <iostream>
#include <assert.h>
#include <cudnn_frontend.h>

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

static inline int getFwdConvPaddedImageDim(int tensorDim, int pad) {
  return tensorDim + (2 * pad);
}

static inline int getFwdConvDilatedFilterDim(int filterDim, int dilation) {
  return ((filterDim - 1) * dilation) + 1;
}

static inline int getFwdConvOutputDim(int tensorDim, int pad, int filterDim, 
                                      int stride, int dilation) {
  int p = (getFwdConvPaddedImageDim(tensorDim, pad) -
           getFwdConvDilatedFilterDim(filterDim, dilation)) / stride + 1;
  return p;
}

static void generateStrides(const int64_t* xdim, int64_t* strideA, int nbDims,
                            cudnnTensorFormat_t filterFormat) {
  if (filterFormat == CUDNN_TENSOR_NCHW ||
      filterFormat == CUDNN_TENSOR_NCHW_VECT_C) {
    strideA[nbDims - 1] = 1;
    for (int64_t d = nbDims - 2; d >= 0; d--) {
      strideA[d] = strideA[d + 1] * xdim[d + 1];
    }
  } else {
    strideA[1]          = 1;
    strideA[nbDims - 1] = strideA[1] * xdim[1];
    for (int64_t d = nbDims - 2; d >= 2; d--) {
      strideA[d] = strideA[d + 1] * xdim[d + 1];
    }
    strideA[0] = strideA[2] * xdim[2];
  }
}

int main(int argc, char const *argv[]) {

  int plan_idx = 0;
  if (argc > 1) {
    plan_idx = atoi(argv[1]);
  }

  int N = 1, C = 4, H = 16, W = 16;
  int K = 12, R = 3, S = 3;

  int64_t conv_pads[] = {0, 0};
  int64_t conv_strides[] = {1, 1};
  int64_t conv_dilations[] = {1, 1};

  cudnnDataType_t dataType = CUDNN_DATA_FLOAT;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;
  cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
  cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;

  const int conv_ndims = 2;
  const int tensor_ndims = 4;
  int64_t x_dims[] = {N, C, H, W};
  int64_t w_dims[] = {K, C, R, S};
  int64_t b_dims[] = {N, K, 1, 1};
  int64_t y_dims[tensor_ndims];
  int64_t x_strides[tensor_ndims];
  int64_t y_strides[tensor_ndims];
  int64_t b_strides[tensor_ndims];
  int64_t w_strides[tensor_ndims];
  y_dims[0] = x_dims[0];
  y_dims[1] = w_dims[0];
  for (int i = 0; i < conv_ndims; i++) {
    y_dims[i + 2] = getFwdConvOutputDim(
                        x_dims[i + 2], conv_pads[i], w_dims[i + 2],
                        conv_strides[i], conv_dilations[i]);
  }
  generateStrides(x_dims, x_strides, tensor_ndims, tensorFormat);
  generateStrides(y_dims, y_strides, tensor_ndims, tensorFormat);
  generateStrides(b_dims, b_strides, tensor_ndims, tensorFormat);
  generateStrides(w_dims, w_strides, tensor_ndims, tensorFormat);

  printf("LOG >>> Input  dims: (%ld, %ld, %ld, %ld)\n",
         x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
  printf("LOG >>> Filter dims: (%ld, %ld, %ld, %ld)\n",
         w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
  printf("LOG >>> Bias   dims: (%ld, %ld, %ld, %ld)\n",
         b_dims[0], b_dims[1], b_dims[2], b_dims[3]);
  printf("LOG >>> Output dims: (%ld, %ld, %ld, %ld)\n",
         y_dims[0], y_dims[1], y_dims[2], y_dims[3]);

  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
  auto tensor_x = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, x_dims)
                      .setStrides(tensor_ndims, x_strides)
                      .setId('x')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_x.get_status());

  auto tensor_y = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, y_dims)
                      .setStrides(tensor_ndims, y_strides)
                      .setId('y')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_y.get_status());

  auto tensor_z = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, y_dims)
                      .setStrides(tensor_ndims, y_strides)
                      .setId('z')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_z.get_status());

  auto tensor_w = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, w_dims)
                      .setStrides(tensor_ndims, w_strides)
                      .setId('w')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_w.get_status());

  auto tensor_b = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, b_dims)
                      .setStrides(tensor_ndims, b_strides)
                      .setId('b')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_b.get_status());

  auto tensor_conv = cudnn_frontend::TensorBuilder()
                         .setDim(tensor_ndims, y_dims)
                         .setStrides(tensor_ndims, y_strides)
	                       .setVirtual()
                         .setId('C')
                         .setAlignment(32)
                         .setDataType(dataType)
                         .build();
  checkCUDNN(tensor_conv.get_status());

  auto tensor_add = cudnn_frontend::TensorBuilder()
                        .setDim(tensor_ndims, y_dims)
                        .setStrides(tensor_ndims, y_strides)
	                      .setVirtual()
                        .setId('A')
                        .setAlignment(32)
                        .setDataType(dataType)
                        .build();
  checkCUDNN(tensor_add.get_status());

  auto tensor_bias = cudnn_frontend::TensorBuilder()
                         .setDim(tensor_ndims, y_dims)
                         .setStrides(tensor_ndims, y_strides)
	                       .setVirtual()
                         .setId('B')
                         .setAlignment(32)
                         .setDataType(dataType)
                         .build();
  checkCUDNN(tensor_bias.get_status());

  auto conv_desc = cudnn_frontend::ConvDescBuilder()
                       .setDataType(computeType)
                       .setMathMode(convMode)
                       .setNDims(conv_ndims)
                       .setStrides(conv_ndims, conv_strides)
                       .setPrePadding(conv_ndims, conv_pads)
                       .setPostPadding(conv_ndims, conv_pads)
                       .setDilation(conv_ndims, conv_dilations)
                       .build();
  checkCUDNN(conv_desc.get_status());

  auto conv_direction = CUDNN_BACKEND_OPERATION_CONVOLUTION_FORWARD_DESCRIPTOR;
  const FLOAT_T alpha = 1.0;
  const FLOAT_T beta = 0.0;
  const FLOAT_T alpha2 = 0.0;
  auto conv_op = cudnn_frontend::OperationBuilder(conv_direction)
                     .setxDesc(tensor_x)
                     .setyDesc(tensor_conv)
                     .setwDesc(tensor_w)
                     .setcDesc(conv_desc)
                     .setAlpha(alpha)
                     .setBeta(beta)
                     .build();
  checkCUDNN(conv_op.get_status());

  auto add_desc = cudnn_frontend::PointWiseDescBuilder()
                      .setMode(CUDNN_POINTWISE_ADD)
                      .setMathPrecision(dataType)
                      .build();
  checkCUDNN(add_desc.get_status());

  auto pointwise_mode =CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
  auto add_op = cudnn_frontend::OperationBuilder(pointwise_mode)
                    .setxDesc(conv_op.getOutputTensor())
                    .setbDesc(tensor_z)
                    .setyDesc(tensor_add)
                    .setpwDesc(add_desc)
                    .setAlpha(alpha)
                    .setAlpha2(alpha2)
                    .build();
  checkCUDNN(add_op.get_status());

  auto bias_desc = cudnn_frontend::PointWiseDescBuilder()
                       .setMode(CUDNN_POINTWISE_ADD)
                       .setMathPrecision(dataType)
                       .build();
  checkCUDNN(bias_desc.get_status());

  auto bias_op = cudnn_frontend::OperationBuilder(pointwise_mode)
                     .setxDesc(add_op.getOutputTensor())
                     .setbDesc(tensor_b)
                     .setyDesc(tensor_bias)
                     .setpwDesc(bias_desc)
                     .build();
  checkCUDNN(bias_op.get_status());

  auto activation_desc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_RELU_FWD)
                             .setMathPrecision(dataType)
                             .build();
  checkCUDNN(activation_desc.get_status());

	auto activation_op = cudnn_frontend::OperationBuilder(pointwise_mode)
                           .setxDesc(bias_op.getOutputTensor())
                           .setyDesc(tensor_y)
                           .setpwDesc(activation_desc)
                           .build();
  checkCUDNN(activation_op.get_status());

	std::array<cudnn_frontend::Operation const*, 4> ops =
      {&conv_op, &add_op, &bias_op, &activation_op};
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  checkCUDNN(op_graph.get_status());

  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(op_graph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();
  auto fallback = cudnn_frontend::EngineFallbackListBuilder()
                      .setOperationGraph(op_graph)
                      .setOperation(conv_direction)
                      .build();
  auto &heuristics_configs = heuristics.getEngineConfig(
                                            heuristics.getEngineConfigCount());
  auto &fallback_configs = fallback.getFallbackList();
  printf("LOG >>> Engines size (heuristics): %ld\n", heuristics_configs.size());
  printf("LOG >>> Engines size (fallback): %ld\n", fallback_configs.size());

  auto filter_fn = [=](cudnnBackendDescriptor_t engine_config) -> bool {
    return cudnn_frontend::hasNumericalNote<
               CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);
  };
  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_frontend::filter(heuristics_configs, filtered_configs, filter_fn);
  cudnn_frontend::filter(fallback_configs, filtered_configs, filter_fn);
  printf("LOG >>> Engines size (filtered): %ld\n", filtered_configs.size());

  std::vector<cudnn_frontend::ExecutionPlan> workable_plans;
  for (int i = 0; i < filtered_configs.size(); i++) {
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnn)
                    .setEngineConfig(filtered_configs[i], op_graph.getTag())
                    .build();
    printf("LOG >>> Building Plan (%s): ", plan.getTag().c_str());
    cudnnStatus_t status = plan.get_status();
    if (status != CUDNN_STATUS_SUCCESS) {
      printf("Fail\n");
      continue;
    } else {
      printf("Success\n");
    }
    workable_plans.push_back(std::move(plan));
  }
  printf("LOG >>> Engines size (workable): %ld\n", workable_plans.size());

  printf("LOG >>> Selecting Engine (%s)\n",
         workable_plans[plan_idx].getTag().c_str());
  auto workspace_bytes = workable_plans[plan_idx].getWorkspaceSize(); 
  printf("LOG >>> Workspace size (bytes): %ld\n", workspace_bytes);

  void* d_workspace{nullptr};
  if (workspace_bytes != 0) {
    checkCUDA(cudaMalloc(&d_workspace, workspace_bytes));
  }

  int x_size = x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3];
  int y_size = y_dims[0] * y_dims[1] * y_dims[2] * y_dims[3];
  int b_size = b_dims[0] * b_dims[1] * b_dims[2] * b_dims[3];
  int w_size = w_dims[0] * w_dims[1] * w_dims[2] * w_dims[3];

  int x_bytes = x_size * sizeof(FLOAT_T);
  int y_bytes = y_size * sizeof(FLOAT_T);
  int b_bytes = b_size * sizeof(FLOAT_T);
  int w_bytes = w_size * sizeof(FLOAT_T);

  FLOAT_T *x;
  FLOAT_T *y;
  FLOAT_T *b;
  FLOAT_T *w;
  checkCUDA(cudaMallocManaged(&x, x_bytes));
  checkCUDA(cudaMallocManaged(&y, y_bytes));
  checkCUDA(cudaMallocManaged(&b, b_bytes));
  checkCUDA(cudaMallocManaged(&w, w_bytes));

  srand(3);
  init_input(x, x_size);
  init_input(y, y_size);
  init_input(b, b_size);
  init_input(w, w_size);

  void * data_ptrs[] = {x, y, w, y, b};
  int64_t uids[] = {'x', 'y', 'w', 'z', 'b'};
  auto variant_pack = cudnn_frontend::VariantPackBuilder()
                          .setWorkspacePointer(d_workspace)
                          .setDataPointers(5, data_ptrs)
                          .setUids(5, uids)
                          .build();
  checkCUDNN(variant_pack.get_status());

  checkCUDNN(cudnnBackendExecute(cudnn, workable_plans[plan_idx].get_raw_desc(),
                                 variant_pack.get_raw_desc()));
  
  print_output(y, y_size, "Y out:", -1);

  checkCUDA(cudaFree(w));
  checkCUDA(cudaFree(b));
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(x));
  if (workspace_bytes != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
}
