#include <cuda_runtime.h>
#include <cudnn.h>
#include <iostream>
#include <cstdio>

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
    cudnnBackendDescriptor_t engine;
    CHECK_CUDNN(cudnnBackendCreateDescriptor(CUDNN_BACKEND_ENGINE_DESCRIPTOR, &engine));
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_OPERATION_GRAPH, CUDNN_TYPE_BACKEND_DESCRIPTOR, 1, &op_graph
    ));
    int64_t gidx = 0;
    CHECK_CUDNN(cudnnBackendSetAttribute(
        engine, CUDNN_ATTR_ENGINE_GLOBAL_INDEX, CUDNN_TYPE_INT64, 1, &gidx
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
