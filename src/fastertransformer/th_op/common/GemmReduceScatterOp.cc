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

#include "src/fastertransformer/th_op/common/GemmReduceScatterOp.h"


namespace torch_ext {

#ifdef BUILD_MULTI_GPU
template<typename T>
ncclDataType_t getNcclDataType()
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
    ncclDataType_t nccl_data_type;
    if (std::is_same<T, float>::value) {
        nccl_data_type = ncclFloat;
    }
    else if (std::is_same<T, half>::value) {
        nccl_data_type = ncclHalf;
    }
#if defined(ENABLE_BF16) && defined(ENABLE_BF16_NCCL)
    else if (std::is_same<T, __nv_bfloat16>::value) {
        nccl_data_type = ncclBfloat16;
    }
#endif
    else if (std::is_same<T, int>::value) {
        nccl_data_type = ncclInt;
    }
    else if (std::is_same<T, char>::value) {
        nccl_data_type = ncclChar;
    }
    else if (std::is_same<T, bool>::value) {
        nccl_data_type = ncclInt8;
    }
    else {
        printf("[ERROR] NCCL only support float, half, bfloat16, int, char, and bool. \n");
        exit(-1);
    }
    return nccl_data_type;
}
#endif

template<typename T>
void myNcclReduceSum(const T* send_buf, T* recv_buf, const int data_size, const int rank, ft::NcclParam nccl_param, cudaStream_t stream)
{
    FT_LOG_DEBUG("%s start", __PRETTY_FUNCTION__);
#ifdef BUILD_MULTI_GPU
    ncclDataType_t nccl_data_type = getNcclDataType<T>();
    NCCLCHECK(ncclGroupStart());
    NCCLCHECK(ncclReduce(
        (const void*)send_buf, (void*)recv_buf, data_size, nccl_data_type, ncclSum, rank, nccl_param.nccl_comm_, stream));
    NCCLCHECK(ncclGroupEnd());
#endif
    FT_LOG_DEBUG("%s stop", __PRETTY_FUNCTION__);
}

GemmReduceScatterOp::GemmReduceScatterOp(int64_t _chunkNum): chunkNum(_chunkNum){
    ft::mpi::initialize(NULL, NULL);
    rank       = ft::mpi::getCommWorldRank();
    world_size = ft::mpi::getCommWorldSize();
    ft::ftNcclInitialize(tensor_para, pipeline_para, world_size, 1);

    streams = new cudaStream_t[streamNum];
    for (int i = 0; i < streamNum; i++) {
        cudaStreamCreate(&streams[i]);
    }
}
GemmReduceScatterOp::~GemmReduceScatterOp(){
    ft::mpi::finalize();

    for (int i = 0; i < streamNum; i++) {
        cudaStreamDestroy(streams[i]);
    }
    delete[] streams;
}

void GemmReduceScatterOp::GemmReduceScatter(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2,
                                    int64_t M, int64_t N, int64_t K)
{
    st_ = m1.scalar_type();

    switch (st_) {
        case at::ScalarType::Float:
            GemmReduceScatterDo<float, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_32F, CUDA_R_32F);
            break;
        case at::ScalarType::Half:
            GemmReduceScatterDo<half, half>(m1, m2, m1m2, __float2half(1.0f), __float2half(0.0f), M, N, K, CUDA_R_16F, CUDA_R_16F);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            GemmReduceScatterDo<__nv_bfloat16, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_16BF, CUDA_R_32F);
            break;
#endif
    }
}


template<typename T1, typename T2>
void GemmReduceScatterOp::GemmReduceScatterDo(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2, const T2& alpha, const T2& beta,
                                    int64_t& M, int64_t& N, int64_t& K, cudaDataType_t Dtype, cudaDataType computeType)
{

    T1 *m1m2_ptr = static_cast<T1 *>(m1m2.data_ptr());
    const T1 *m1_ptr = static_cast<const T1 *>(m1.data_ptr());
    const T1 *m2_ptr = static_cast<const T1 *>(m2.data_ptr());

    size_t rank_size = M * N / world_size;
    size_t mr_start = rank * rank_size;
    size_t mr_end = (rank + 1) * rank_size;

    size_t chunkSize = (M * N) / chunkNum;
    size_t curStream = 0;

    for(size_t curPos = 0; curPos < M*N; curPos += chunkSize) {
        cublasProcessMatMulChunk(streams[curStream], curPos, curPos + chunkSize,
                                 static_cast<const T2 *>(&alpha),
                                 static_cast<const T2 *>(&beta),
                                 m1_ptr, m2_ptr, m1m2_ptr, M, N, K, Dtype, computeType);
        myNcclReduceSum(m1m2_ptr+curPos, m1m2_ptr+curPos, chunkSize, curPos/rank_size, tensor_para, streams[curStream]);
        curStream = (++curStream) >= streamNum ? 0 : curStream;
    }
    waitOnStreams(streams, streamNum);
}

template<typename T1, typename T2>
void GemmReduceScatterOp::cublasProcessMatMulChunk(cudaStream_t stream,
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

static auto fasterTransformerGemmReduceScatterTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GemmReduceScatterOp>("FasterTransformerGemmReduceScatterOp")
#else
    torch::jit::class_<torch_ext::GemmReduceScatterOp>("FasterTransformer", "GemmReduceScatterOp")
#endif
        .def(torch::jit::init<int64_t>())
        .def("GemmReduceScatter", &torch_ext::GemmReduceScatterOp::GemmReduceScatter);
    