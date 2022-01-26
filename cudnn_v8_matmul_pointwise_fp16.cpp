#include <iostream>
#include <cudnn_frontend.h>
#include <cuda_fp16.h>
#include <time.h>
#include <sys/time.h>

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

uint64_t CpuTimer() {
  timeval tv;
  gettimeofday(&tv, 0);
  return (tv.tv_sec * 1000) + (tv.tv_usec / 1000);
}

void init_input(FLOAT_T *ptr, int size) {
  FLOAT_T* ptr_host = new FLOAT_T[size];
  for (int i = 0; i < size; i++) {
    float val = static_cast<float>(rand()) / RAND_MAX;
    ptr_host[i]  = static_cast<FLOAT_T>(val);
  }
  checkCUDA(cudaMemcpy(ptr, ptr_host, sizeof(FLOAT_T) * size,
                       cudaMemcpyHostToDevice));
  delete[] ptr_host;
}

void print_output(const FLOAT_T* ptr, int size, const char* message,
                  int lines = 10) {
  checkCUDA(cudaDeviceSynchronize());
  FLOAT_T* ptr_host = new FLOAT_T[size];
  checkCUDA(cudaMemcpy(ptr_host, ptr, sizeof(FLOAT_T) * size,
                       cudaMemcpyDeviceToHost));

  const int num_per_line = 20;
  int limit = INT_MAX;
  if (lines != -1) {
    limit = lines * num_per_line;
  }

  printf("%s (showing %d elements):\n", message, std::min(size, limit));
  for (int i = 0; i < std::min(size, limit); i++) {
    printf("%f, ", static_cast<float>(ptr_host[i]));
    if ((i + 1) % num_per_line == 0) {
      printf("\n");
    }
  }
  printf("\n");
  delete[] ptr_host;
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

int main(int argc, char **argv) {
  int plan_idx = 0;
  if (argc > 1) {
    plan_idx = atoi(argv[1]);
  }

  const int M = 64;
  const int K = 32;
  const int N = 64;
  const int ndims = 3;
  int64_t a_dims[] = {1, M, K};
  int64_t b_dims[] = {1, K, N};
  int64_t c_dims[] = {1, M, N};
  int64_t z_dims[] = {1, 1, N};
  int64_t a_strides[ndims];
  int64_t b_strides[ndims];
  int64_t c_strides[ndims];
  int64_t z_strides[ndims];

  cudnnDataType_t dataType = CUDNN_DATA_HALF;
  cudnnDataType_t computeType = CUDNN_DATA_FLOAT;

	generateStrides(a_dims, a_strides, ndims, CUDNN_TENSOR_NCHW);
	generateStrides(b_dims, b_strides, ndims, CUDNN_TENSOR_NCHW);
	generateStrides(c_dims, c_strides, ndims, CUDNN_TENSOR_NCHW);
	generateStrides(z_dims, z_strides, ndims, CUDNN_TENSOR_NCHW);
  
  auto tensor_a = cudnn_frontend::TensorBuilder()
                      .setDim(ndims, a_dims)
                      .setStrides(ndims, a_strides)
                      .setId('a')
                      .setAlignment(16)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_a.get_status());

  auto tensor_b = cudnn_frontend::TensorBuilder()
                      .setDim(ndims, b_dims)
                      .setStrides(ndims, b_strides)
                      .setId('b')
                      .setAlignment(16)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_b.get_status());

  auto tensor_z = cudnn_frontend::TensorBuilder()
                      .setDim(ndims, z_dims)
                      .setStrides(ndims, z_strides)
                      .setId('z')
                      .setAlignment(16)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_z.get_status());

  auto vtensor_matmul = cudnn_frontend::TensorBuilder()
                            .setDim(ndims, c_dims)
                            .setStrides(ndims, c_strides)
                            .setId('A')
                            .setAlignment(16)
                            .setVirtual()
                            .setDataType(dataType)
                            .build();
  checkCUDNN(vtensor_matmul.get_status());

  auto vtensor_bias = cudnn_frontend::TensorBuilder()
                          .setDim(ndims, c_dims)
                          .setStrides(ndims, c_strides)
                          .setId('B')
                          .setAlignment(16)
                          .setVirtual()
                          .setDataType(dataType)
                          .build();
  checkCUDNN(vtensor_bias.get_status());

  auto tensor_c = cudnn_frontend::TensorBuilder()
                      .setDim(ndims, c_dims)
                      .setStrides(ndims, c_strides)
                      .setId('c')
                      .setAlignment(16)
                      .setDataType(dataType)
                      .build();
  checkCUDNN(tensor_c.get_status());


  auto matmul_desc = cudnn_frontend::MatMulDescBuilder()
                         .setMathPrecision(computeType)
                         .build();
  checkCUDNN(matmul_desc.get_status());

  auto matmul_mode = CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR;
  auto matmul_op = cudnn_frontend::OperationBuilder(matmul_mode)
                       .setaMatDesc(tensor_a)
                       .setbMatDesc(tensor_b)
                       .setcMatDesc(vtensor_matmul)
                       .setmatmulDesc(matmul_desc)
                       .build();
  checkCUDNN(matmul_op.get_status());

  auto bias_desc = cudnn_frontend::PointWiseDescBuilder()
                       .setMode(CUDNN_POINTWISE_ADD)
                       .setMathPrecision(computeType)
                       .build();
  checkCUDNN(bias_desc.get_status());
 
  auto pointwise_mode = CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR;
  auto bias_op = cudnn_frontend::OperationBuilder(pointwise_mode)
                     .setxDesc(matmul_op.getOutputTensor())
                     .setbDesc(tensor_z)
                     .setyDesc(vtensor_bias)
                     .setpwDesc(bias_desc)
                     .build();
  checkCUDNN(bias_op.get_status());

  auto activation_desc = cudnn_frontend::PointWiseDescBuilder()
                             .setMode(CUDNN_POINTWISE_GELU_FWD)
                             .setMathPrecision(computeType)
                             .build();
  checkCUDNN(activation_desc.get_status());

  auto activation_op = cudnn_frontend::OperationBuilder(pointwise_mode)
                           .setxDesc(bias_op.getOutputTensor())
                           .setyDesc(tensor_c)
                           .setpwDesc(activation_desc)
                           .build();
  checkCUDNN(activation_op.get_status());

  std::array<cudnn_frontend::Operation const*, 3> ops = {
      &matmul_op, &bias_op, &activation_op};

  cudnnHandle_t cudnn;
	checkCUDNN(cudnnCreate(&cudnn));
  auto op_graph = cudnn_frontend::OperationGraphBuilder()
                      .setHandle(cudnn)
                      .setOperationGraph(ops.size(), ops.data())
                      .build();
  checkCUDNN(op_graph.get_status());

  // Only heuristics is supported at this moment.
  auto heuristics = cudnn_frontend::EngineHeuristicsBuilder()
                        .setOperationGraph(op_graph)
                        .setHeurMode(CUDNN_HEUR_MODE_INSTANT)
                        .build();
  auto &heuristics_configs = heuristics.getEngineConfig(
                                            heuristics.getEngineConfigCount());
  printf("LOG >>> Engines size (heuristics): %ld\n", heuristics_configs.size());

  auto filter_fn = [=](cudnnBackendDescriptor_t engine_config) -> bool {
    return cudnn_frontend::hasNumericalNote<
               CUDNN_NUMERICAL_NOTE_DOWN_CONVERT_INPUTS>(engine_config);
  };
  cudnn_frontend::EngineConfigList filtered_configs;
  cudnn_frontend::filter(heuristics_configs, filtered_configs, filter_fn);
  printf("LOG >>> Engines size (filtered): %ld\n", filtered_configs.size());

  std::vector<cudnn_frontend::ExecutionPlan> workable_plans;
  for (int i = 0; i < filtered_configs.size(); i++) {
    // Since the kernels are generated during the runtime, we apply a timer over
    // the plan building time.
    uint64_t t0 = CpuTimer();
    auto plan = cudnn_frontend::ExecutionPlanBuilder()
                    .setHandle(cudnn)
                    .setEngineConfig(filtered_configs[i], op_graph.getTag())
                    .build();
    uint64_t t1 = CpuTimer();
    printf("LOG >>> Building Plan (%s): ", plan.getTag().c_str());
    cudnnStatus_t status = plan.get_status();
    if (status != CUDNN_STATUS_SUCCESS) {
      printf("Fail\n");
      continue;
    } else {
      printf("Success (Time(ms)=%lu)\n", t1 - t0);
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

  int a_size = a_dims[0] * a_dims[1] * a_dims[2];
  int b_size = b_dims[0] * b_dims[1] * b_dims[2];
  int z_size = z_dims[0] * z_dims[1] * z_dims[2];
  int c_size = c_dims[0] * c_dims[1] * c_dims[2];

  int a_bytes = a_size * sizeof(FLOAT_T);
  int b_bytes = b_size * sizeof(FLOAT_T);
  int z_bytes = z_size * sizeof(FLOAT_T);
  int c_bytes = c_size * sizeof(FLOAT_T);

  FLOAT_T *a;
  FLOAT_T *b;
  FLOAT_T *z;
  FLOAT_T *c;
  checkCUDA(cudaMalloc(&a, a_bytes));
  checkCUDA(cudaMalloc(&b, b_bytes));
  checkCUDA(cudaMalloc(&z, z_bytes));
  checkCUDA(cudaMalloc(&c, c_bytes));

  srand(3);
  init_input(a, a_size);
  init_input(b, b_size);
  init_input(z, z_size);

  void* data_ptrs[] = {a, b, c, z};
  int64_t uids[] = {'a', 'b', 'c', 'z'};
  auto variant_pack = cudnn_frontend::VariantPackBuilder()
                          .setWorkspacePointer(d_workspace)
                          .setDataPointers(4, data_ptrs)
                          .setUids(4, uids)
                          .build();
  checkCUDNN(variant_pack.get_status());

  checkCUDNN(cudnnBackendExecute(cudnn, workable_plans[plan_idx].get_raw_desc(),
                                 variant_pack.get_raw_desc()));

  print_output(c, c_size, "c out:", 1);

  checkCUDA(cudaFree(a));
  checkCUDA(cudaFree(b));
  checkCUDA(cudaFree(c));
  checkCUDA(cudaFree(z));
  if (workspace_bytes != 0) {
    checkCUDA(cudaFree(d_workspace));
  }
}
