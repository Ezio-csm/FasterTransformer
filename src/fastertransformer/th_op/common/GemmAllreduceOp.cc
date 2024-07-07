/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "src/fastertransformer/th_op/common/GemmAllreduceOp.h"


namespace torch_ext {

GemmAllreduceOp::GemmAllreduceOp(int64_t _chunkNum): chunkNum(_chunkNum){
    ft::mpi::initialize(NULL, NULL);
    rank       = ft::mpi::getCommWorldRank();
    world_size = ft::mpi::getCommWorldSize();
    ft::ftNcclInitialize(tensor_para, pipeline_para, world_size, 1);

    streams = new cudaStream_t[streamNum];
    for (int i = 0; i < streamNum; i++) {
        cudaStreamCreate(&streams[i]);
    }
    cudaEventCreateWithFlags(&event_, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&event1_, cudaEventDisableTiming);
}
GemmAllreduceOp::~GemmAllreduceOp(){
    ft::mpi::finalize();

    cudaEventDestroy(event_);
    cudaEventDestroy(event1_);

    for (int i = 0; i < streamNum; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

void GemmAllreduceOp::GemmAllreduce(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2,
                                    int64_t M, int64_t N, int64_t K)
{
    st_ = m1.scalar_type();

    switch (st_) {
        case at::ScalarType::Float:
            GemmAllreduceDo<float, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_32F, CUDA_R_32F);
            break;
        case at::ScalarType::Half:
            GemmAllreduceDo<half, half>(m1, m2, m1m2, __float2half(1.0f), __float2half(0.0f), M, N, K, CUDA_R_16F, CUDA_R_16F);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            GemmAllreduceDo<__nv_bfloat16, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_16BF, CUDA_R_32F);
            break;
#endif
    }
}

// TODO: performance optimization
template<typename T1, typename T2>
void GemmAllreduceOp::GemmAllreduceDo(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2, const T2& alpha, const T2& beta,
                                    int64_t& M, int64_t& N, int64_t& K, cudaDataType_t Dtype, cudaDataType computeType)
{

    T1 *m1m2_ptr = static_cast<T1 *>(m1m2.data_ptr());
    const T1 *m1_ptr = static_cast<const T1 *>(m1.data_ptr());
    const T1 *m2_ptr = static_cast<const T1 *>(m2.data_ptr());

    size_t chunkSize = (M * N) / chunkNum;
    size_t curStream = 0;

    cublasProcessMatMulChunk(streams[0], 0, 0 + chunkSize, 
                                 static_cast<const T2 *>(&alpha),
                                 static_cast<const T2 *>(&beta),
                                 m1_ptr, m2_ptr, m1m2_ptr, M, N, K, Dtype, computeType);
    cudaEventRecord(event_, streams[0]);
    cudaStreamWaitEvent(streams[1], event_);
    // cudaStreamSynchronize(streams[0]);
    ft::ftNcclAllReduceSum(m1m2_ptr+0, m1m2_ptr+0, chunkSize, tensor_para, streams[1]);

    for(size_t curPos = chunkSize; curPos < M*N; curPos += chunkSize) {
        cublasProcessMatMulChunk(streams[0], curPos, curPos + chunkSize, 
                                 static_cast<const T2 *>(&alpha),
                                 static_cast<const T2 *>(&beta),
                                 m1_ptr, m2_ptr, m1m2_ptr, M, N, K, Dtype, computeType);
        cudaEventRecord(event_, streams[0]);
        cudaStreamWaitEvent(streams[1], event_);
        // waitOnStreams(streams, streamNum);
        ft::ftNcclAllReduceSum(m1m2_ptr+curPos, m1m2_ptr+curPos, chunkSize, tensor_para, streams[1]);
        // curStream = (++curStream) >= streamNum ? 0 : curStream;
    }
    cudaEventRecord(event1_, streams[1]);
    cudaStreamWaitEvent(streams[0], event1_);
    // waitOnStreams(streams, streamNum);
}

template<typename T1, typename T2>
void GemmAllreduceOp::cublasProcessMatMulChunk(cudaStream_t stream,
                              size_t chunkStart, size_t chunkEnd,
                              const T2* alpha, const T2* beta,
                              const T1* m1, const T1* m2, T1* m1m2,
                              int64_t M, int64_t N, int64_t K, cudaDataType_t Dtype, cudaDataType computeType)
{
    int rows     = (chunkEnd-chunkStart) / N;
    int startRow = chunkStart / N;

    cublasHandle_t handle = at::cuda::getCurrentCUDABlasHandle();
    cublasSetStream(handle, stream);

    TORCH_CUDABLAS_CHECK(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                                      N, rows, K,
                                      alpha,
                                      m2, Dtype, N,
                                      m1 + startRow * K, Dtype, K,
                                      beta,
                                      m1m2 + startRow * N, Dtype, N,
                                      computeType,
                                      CUBLAS_GEMM_DEFAULT_TENSOR_OP));
}
}  // namespace torch_ext

static auto fasterTransformerGemmAllreduceTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GemmAllreduceOp>("FasterTransformerGemmAllreduceOp")
#else
    torch::jit::class_<torch_ext::GemmAllreduceOp>("FasterTransformer", "GemmAllreduceOp")
#endif
        .def(torch::jit::init<int64_t>())
        .def("GemmAllreduce", &torch_ext::GemmAllreduceOp::GemmAllreduce);
    