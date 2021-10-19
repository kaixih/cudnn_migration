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

  int N = 7, C = 64, H = 21, W = 21;
  int K = 32, R = 3, S = 3;

  int64_t conv_pads[] = {0, 0};
  int64_t conv_strides[] = {1, 1};
  int64_t conv_dilations[] = {1, 1};

  cudnnDataType_t dataType = CUDNN_DATA_INT8;
  cudnnDataType_t computeType = CUDNN_DATA_INT32;
  cudnnTensorFormat_t tensorFormat = CUDNN_TENSOR_NCHW;
  cudnnConvolutionMode_t convMode = CUDNN_CROSS_CORRELATION;

  int64_t vector_cnt = 32;
  int64_t vector_dim = 1;
  const int conv_ndims = 2;
  const int tensor_ndims = 4;
  int64_t x_dims[] = {N, C / vector_cnt, H, W};
  int64_t w_dims[] = {K, C / vector_cnt, R, S};
  int64_t y_dims[tensor_ndims];
  int64_t x_strides[tensor_ndims];
  int64_t y_strides[tensor_ndims];
  int64_t w_strides[tensor_ndims];
  y_dims[0] = x_dims[0];
  y_dims[1] = w_dims[0] / vector_cnt;
  for (int i = 0; i < conv_ndims; i++) {
    y_dims[i + 2] = getFwdConvOutputDim(
                        x_dims[i + 2], conv_pads[i], w_dims[i + 2],
                        conv_strides[i], conv_dilations[i]);
  }
  generateStrides(x_dims, x_strides, tensor_ndims, tensorFormat);
  generateStrides(y_dims, y_strides, tensor_ndims, tensorFormat);
  generateStrides(w_dims, w_strides, tensor_ndims, tensorFormat);

  printf("LOG >>> Input  dims (resized): (%ld, %ld, %ld, %ld)\n",
         x_dims[0], x_dims[1], x_dims[2], x_dims[3]);
  printf("LOG >>> Filter dims (resized): (%ld, %ld, %ld, %ld)\n",
         w_dims[0], w_dims[1], w_dims[2], w_dims[3]);
  printf("LOG >>> Output dims (resized): (%ld, %ld, %ld, %ld)\n",
         y_dims[0], y_dims[1], y_dims[2], y_dims[3]);

  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
  auto tensor_x = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, x_dims)
                      .setStrides(tensor_ndims, x_strides)
                      .setId('x')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .setVectorCountAndDimension(vector_cnt, vector_dim)
                      .build();
  checkCUDNN(tensor_x.get_status());

  auto tensor_y = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, y_dims)
                      .setStrides(tensor_ndims, y_strides)
                      .setId('y')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .setVectorCountAndDimension(vector_cnt, vector_dim)
                      .build();
  checkCUDNN(tensor_y.get_status());

  auto tensor_w = cudnn_frontend::TensorBuilder()
                      .setDim(tensor_ndims, w_dims)
                      .setStrides(tensor_ndims, w_strides)
                      .setId('w')
                      .setAlignment(32)
                      .setDataType(dataType)
                      .setVectorCountAndDimension(vector_cnt, vector_dim)
#if (CUDNN_VERSION >= 8300)
                      .setReorderType(CUDNN_TENSOR_REORDERING_INT8x32)
#endif
                      .build();
  checkCUDNN(tensor_w.get_status());

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
  const float alpha = 1.0;
  const float beta = 0.0;
  auto conv_op = cudnn_frontend::OperationBuilder(conv_direction)
                     .setxDesc(tensor_x)
                     .setyDesc(tensor_y)
                     .setwDesc(tensor_w)
                     .setcDesc(conv_desc)
                     .setAlpha(alpha)
                     .setBeta(beta)
                     .build();
  checkCUDNN(conv_op.get_status());

  std::array<cudnn_frontend::Operation const *, 1> ops = {&conv_op};
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

#define MANUAL_REORDER
#ifdef MANUAL_REORDER
  auto json_handle = json::parse(R"(
                         { "version" : 1, 
                           "rules"   : 
                             [ 
                                 { "rule_id"             : "ConvFwd_eng0", 
                                   "operation"           : "ConvFwd",
                                   "engine"              : 0, 
                                   "knob"                : [],
                                   "cudnn_version_start" : 8000, 
                                   "cudnn_version_end"   : 8300 
                                 }
                             ] 
                         })");
  auto check_int8x32 = [](cudnnDataType_t type, int vector_count) {
      return type == CUDNN_DATA_INT8 && vector_count == 32;
    };
  auto fn = std::bind(check_int8x32, dataType, vector_cnt);
#endif

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
#ifdef MANUAL_REORDER
      if (cudnn_frontend::check_errata(json_handle, plan.getTag(), cudnn, fn)) {
        printf("Fail (int8x32 not supported)\n");
        continue;
      }
#endif
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

  int x_size = vector_cnt * x_dims[0] * x_dims[1] * x_dims[2] * x_dims[3];
  int y_size = vector_cnt * y_dims[0] * y_dims[1] * y_dims[2] * y_dims[3];
  int w_size = vector_cnt * w_dims[0] * w_dims[1] * w_dims[2] * w_dims[3];

  int x_bytes = x_size * sizeof(INT_T);
  int y_bytes = y_size * sizeof(INT_T);
  int w_bytes = w_size * sizeof(INT_T);

  INT_T *x;
  INT_T *y;
  INT_T *w;
  checkCUDA(cudaMallocManaged(&x, x_bytes));
  checkCUDA(cudaMallocManaged(&y, y_bytes));
  checkCUDA(cudaMallocManaged(&w, w_bytes));

  
  srand(3);
  init_input(x, x_size);
  init_input(y, y_size);
  init_input(w, w_size);

#ifdef MANUAL_REORDER
  INT_T *reordered_w;
  checkCUDA(cudaMallocManaged(&reordered_w, w_bytes));
  auto reorder_status = cudnn_frontend::cudnnReorderFilterAndBiasInt8x32(
      cudnn, tensor_w, conv_desc, w, reordered_w, nullptr, nullptr);
  checkCUDNN(reorder_status);
#endif

#ifdef MANUAL_REORDER
  void * data_ptrs[] = {x, y, reordered_w};
#else
  void * data_ptrs[] = {x, y, w};
#endif
  int64_t uids[] = {'x', 'y', 'w'};
  auto variant_pack = cudnn_frontend::VariantPackBuilder()
                          .setWorkspacePointer(d_workspace)
                          .setDataPointers(3, data_ptrs)
                          .setUids(3, uids)
                          .build();
  checkCUDNN(variant_pack.get_status());

  checkCUDNN(cudnnBackendExecute(cudnn, workable_plans[plan_idx].get_raw_desc(),
                                 variant_pack.get_raw_desc()));
  
  print_output(y, y_size, "Y out:", -1);

  checkCUDA(cudaFree(w));
#ifdef MANUAL_REORDER
  checkCUDA(cudaFree(reordered_w));
#endif
  checkCUDA(cudaFree(y));
  checkCUDA(cudaFree(x));
  if (workspace_bytes != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
}
