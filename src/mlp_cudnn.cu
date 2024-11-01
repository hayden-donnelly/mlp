#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdint>
#include <cstdio>
#include "mnist.hpp"

#define CHECK_CUDNN(expression) \
{ \
    cudnnStatus_t status = (expression); \
    if(status != CUDNN_STATUS_SUCCESS) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
            << cudnnGetErrorString(status) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

#define CHECK_CUDA(expression) \
{ \
    cudaError_t error = (expression); \
    if(error != 0) \
    { \
        std::cerr << "Error on line " << __LINE__ << ": " \
            << cudaGetErrorString(error) << std::endl; \
        std::exit(EXIT_FAILURE); \
    } \
}

struct fc_layer_t
{
    cudnnBackendDescriptor_t matmul_desc;
    cudnnBackendDescriptor_t matmul_op_desc;
    cudnnBackendDescriptor_t weight_desc;
    cudnnBackendDescriptor_t output_desc;
};

void tensor_2d_create(
    int64_t dim0, int64_t dim1, int64_t* tensor_count, cudnnBackendDescriptor_t* desc
){
    int64_t n_dims = 2;
    int64_t shape[] = {dim0, dim1};
    int64_t strides[] = {dim1, 1};
    int64_t alignment = 16;
    int64_t uid = (*tensor_count)++;;

    cudnnDataType_t data_type = CUDNN_DATA_FLOAT;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_TENSOR_DESCRIPTOR, desc));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        *desc, CUDNN_ATTR_TENSOR_DATA_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &data_type
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        *desc, CUDNN_ATTR_TENSOR_DIMENSIONS, CUDNN_TYPE_INT64, n_dims, shape
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        *desc, CUDNN_ATTR_TENSOR_STRIDES, CUDNN_TYPE_INT64, n_dims, strides
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        *desc, CUDNN_ATTR_TENSOR_BYTE_ALIGNMENT, CUDNN_TYPE_INT64, 1, &alignment
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        *desc, CUDNN_ATTR_TENSOR_UNIQUE_ID, CUDNN_TYPE_INT64, 1, &uid
    ));
    CHECK_CUDNN(cudnnBackendFinalize(*desc));
}

void fc_layer_create(
    int64_t batch_size, int64_t input_dim, int64_t output_dim, int64_t* tensor_count, 
    fc_layer_t* fc, cudnnBackendDescriptor_t* input_desc
){
    tensor_2d_create(batch_size, output_dim, tensor_count, &fc->output_desc);
    tensor_2d_create(input_dim, output_dim, tensor_count, &fc->weight_desc);
    
    cudnnDataType_t comp_type = CUDNN_DATA_FLOAT;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_MATMUL_DESCRIPTOR, &fc->matmul_desc));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        fc->matmul_desc, CUDNN_ATTR_MATMUL_COMP_TYPE, CUDNN_TYPE_DATA_TYPE, 1, &comp_type
    ));
    CHECK_CUDNN(cudnnBackendFinalize(fc->matmul_desc));

    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_MATMUL_DESCRIPTOR, &fc->matmul_op_desc));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        fc->matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_ADESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, input_desc 
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        fc->matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_BDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fc->weight_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        fc->matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_CDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fc->output_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        fc->matmul_op_desc, CUDNN_ATTR_OPERATION_MATMUL_DESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fc->matmul_desc
    ));
    CHECK_CUDNN(cudnnBackendFinalize(fc->matmul_op_desc));
}

int main_old()
{
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    printf("Initialized cuDNN\n");
    printf("cuDNN version: %zu\n", cudnnGetVersion());

    load_mnist();
    printf("Loaded MNIST\n");
    print_image(train_image[2]);

    constexpr int64_t input_dim = 784;
    constexpr int64_t hidden_dim = 256;
    constexpr int64_t output_dim = 10;
    constexpr int64_t batch_size = 32;

    float* d_fc1_weights;
    float* d_fc2_weights;
    float* d_fc1_bias;
    float* d_fc2_bias;
    float* d_input;
    float* d_hidden;
    float* d_output;

    CHECK_CUDA(cudaMalloc(&d_fc1_weights, sizeof(float) * input_dim * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_fc2_weights, sizeof(float) * hidden_dim * output_dim));
    CHECK_CUDA(cudaMalloc(&d_fc1_bias, sizeof(float) * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_fc2_bias, sizeof(float) * output_dim));
    CHECK_CUDA(cudaMalloc(&d_input, sizeof(float) * batch_size * input_dim));
    CHECK_CUDA(cudaMalloc(&d_hidden, sizeof(float) * batch_size * hidden_dim));
    CHECK_CUDA(cudaMalloc(&d_output, sizeof(float) * batch_size * output_dim));

    // Create graph ops and tensors.
    int64_t tensor_count = 0;
    cudnnBackendDescriptor_t input_desc;
    tensor_2d_create(batch_size, input_dim, &tensor_count, &input_desc);
    printf("Created input tensor\ntensor_count: %ld\n", tensor_count);
    fc_layer_t fc1;
    fc_layer_create(batch_size, input_dim, hidden_dim, &tensor_count, &fc1, &input_desc);
    printf("Created fc1\ntensor_count: %ld\n", tensor_count);
    /*fc_layer_t fc2;
    fc_layer_create(batch_size, hidden_dim, output_dim, &tensor_count, &fc2, &fc1.output_desc); 
    printf("Created fc2\ntensor_count: %ld\n", tensor_count);
    printf("Final tensor_count: %ld\n", tensor_count);*/
    
    // Create op graph.
    //cudnnBackendDescriptor_t ops[] = {fc1.matmul_op_desc, fc2.matmul_op_desc};
    cudnnBackendDescriptor_t op_graph;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &fc1.matmul_op_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn
    ));
    CHECK_CUDNN(cudnnBackendFinalize(op_graph));
    printf("Created graph\n");
    
    // Create engine.
    cudnnBackendBehaviorNote_t behavior = CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    cudnnBackendDescriptor_t engine;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, 1, &behavior
    ));
    int64_t gid = 0;
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gid
    ));
    CHECK_CUDNN(cudnnBackendFinalize(engine));
    printf("Created engine\n");
    
    // Create engine config.
    cudnnBackendDescriptor_t engine_cfg;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engine_cfg));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine_cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine
    ));
    CHECK_CUDNN(cudnnBackendFinalize(engine_cfg));
    int64_t workspace_size;
    CHECK_CUDNN(cudnnBackendGetAttribute(
        engine_cfg, CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &workspace_size
    ));
    
    CHECK_CUDNN(cudnnDestroy(cudnn));
}

int main()
{
    cudnnHandle_t cudnn;
    CHECK_CUDNN(cudnnCreate(&cudnn));
    printf("Initialized cuDNN\n");
    printf("cuDNN version: %zu\n", cudnnGetVersion());

    int64_t tensor_count = 0;
    cudnnBackendDescriptor_t input_desc;
    tensor_2d_create(1, 32, &tensor_count, &input_desc);
    
    cudnnBackendDescriptor_t output_desc;
    tensor_2d_create(1, 32, &tensor_count, &output_desc);

    cudnnPointwiseMode_t act_mode = CUDNN_POINTWISE_RELU_FWD;
    cudnnDataType_t act_data_type = CUDNN_DATA_FLOAT;
    cudnnBackendDescriptor_t relu_desc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_POINTWISE_DESCRIPTOR, &relu_desc));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        relu_desc, CUDNN_ATTR_POINTWISE_MODE, CUDNN_TYPE_POINTWISE_MODE, 1, &act_mode
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        relu_desc, CUDNN_ATTR_POINTWISE_MATH_PREC, CUDNN_TYPE_DATA_TYPE, 1, &act_data_type
    ));
    CHECK_CUDNN(cudnnBackendFinalize(relu_desc));

    cudnnBackendDescriptor_t relu_op_desc;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATION_POINTWISE_DESCRIPTOR, &relu_op_desc));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        relu_op_desc, CUDNN_ATTR_OPERATION_POINTWISE_PW_DESCRIPTOR, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        relu_op_desc, CUDNN_ATTR_OPERATION_POINTWISE_XDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &input_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        relu_op_desc, CUDNN_ATTR_OPERATION_POINTWISE_YDESC, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &output_desc
    ));
    CHECK_CUDNN(cudnnBackendFinalize(relu_op_desc));
    printf("Final tensor_count: %ld\n", tensor_count);

    // Create op graph.
    cudnnBackendDescriptor_t op_graph;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_OPERATIONGRAPH_DESCRIPTOR, &op_graph));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op_graph, CUDNN_ATTR_OPERATIONGRAPH_OPS, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &relu_op_desc
    ));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        op_graph, CUDNN_ATTR_OPERATIONGRAPH_HANDLE, CUDNN_TYPE_HANDLE, 1, &cudnn
    ));
    CHECK_CUDNN(cudnnBackendFinalize(op_graph));
    printf("Created graph\n");

    // Create engine.
    //cudnnBackendBehaviorNote_t behavior = CUDNN_BEHAVIOR_NOTE_RUNTIME_COMPILATION;
    cudnnBackendDescriptor_t engine;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph
    ));
    //CHECK_CUDNN(cudnnBackendSetAttribute(
    //    engine, CUDNN_ATTR_ENGINE_BEHAVIOR_NOTE, CUDNN_TYPE_BEHAVIOR_NOTE, 1, &behavior
    //));
    int64_t gid = 0;
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gid
    ));
    CHECK_CUDNN(cudnnBackendFinalize(engine));
    printf("Created engine\n");

    // Create engine config.
    cudnnBackendDescriptor_t engine_cfg;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINECFG_DESCRIPTOR, &engine_cfg));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine_cfg, CUDNN_ATTR_ENGINECFG_ENGINE, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &engine
    ));
    CHECK_CUDNN(cudnnBackendFinalize(engine_cfg));
    int64_t workspace_size;
    CHECK_CUDNN(cudnnBackendGetAttribute(
        engine_cfg, CUDNN_ATTR_ENGINECFG_WORKSPACE_SIZE, CUDNN_TYPE_INT64, 1, NULL, &workspace_size
    ));
    printf("Created engine config\n");

    CHECK_CUDNN(cudnnDestroy(cudnn));
}
