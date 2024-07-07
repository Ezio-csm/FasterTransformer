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

#include "src/fastertransformer/th_op/common/GemmAllgatherOp.h"


namespace torch_ext {

GemmAllgatherOp::GemmAllgatherOp(int64_t _chunkNum): chunkNum(_chunkNum){
    ft::mpi::initialize(NULL, NULL);
    rank       = ft::mpi::getCommWorldRank();
    world_size = ft::mpi::getCommWorldSize();
    ft::ftNcclInitialize(tensor_para, pipeline_para, world_size, 1);

    streams = new cudaStream_t[streamNum];
    for (int i = 0; i < streamNum; i++) {
        cudaStreamCreate(&streams[i]);
        // cudaEventCreate(&events[i]);
    }
}
GemmAllgatherOp::~GemmAllgatherOp(){
    ft::mpi::finalize();

    for (int i = 0; i < streamNum; i++) {
        cudaStreamDestroy(streams[i]);
        // cudaEventDestroy(events[i]);
    }
    delete[] streams;
}

void GemmAllgatherOp::GemmAllgather(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2,
                                    int64_t M, int64_t N, int64_t K)
{
    st_ = m1.scalar_type();

    switch (st_) {
        case at::ScalarType::Float:
            GemmAllgatherDo<float, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_32F, CUDA_R_32F);
            break;
        case at::ScalarType::Half:
            GemmAllgatherDo<half, half>(m1, m2, m1m2, __float2half(1.0f), __float2half(0.0f), M, N, K, CUDA_R_16F, CUDA_R_16F);
            break;
#ifdef ENABLE_BF16
        case at::ScalarType::BFloat16:
            GemmAllgatherDo<__nv_bfloat16, float>(m1, m2, m1m2, 1.0f, 0.0f, M, N, K, CUDA_R_16BF, CUDA_R_32F);
            break;
#endif
    }
}


template<typename T1, typename T2>
void GemmAllgatherOp::GemmAllgatherDo(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2, const T2& alpha, const T2& beta,
                                    int64_t& M, int64_t& N, int64_t& K, cudaDataType_t Dtype, cudaDataType computeType)
{

    T1 *m1m2_ptr = static_cast<T1 *>(m1m2.data_ptr());
    const T1 *m1_ptr = static_cast<const T1 *>(m1.data_ptr());
    const T1 *m2_ptr = static_cast<const T1 *>(m2.data_ptr());
    
    auto tmp_tensor = th::empty_like(m1);
    T1 *recv_buf = static_cast<T1 *>(tmp_tensor.data_ptr());

    size_t m1_chunkSize = M*K/world_size;

    size_t chunkSize = (M * N) / world_size;
    size_t curStream = 0;// 0,1 for recv&calc, 2 for send

    cublasProcessMatMulChunk(streams[curStream], rank*chunkSize, (rank+1)*chunkSize, 
                            static_cast<const T2 *>(&alpha),
                            static_cast<const T2 *>(&beta),
                            m1_ptr, m2_ptr, m1m2_ptr, M/world_size, N, K, Dtype, computeType);
    curStream ^= 1;

    for(int dt = 1; dt < world_size; ++dt){
        int curRank = ring_bck(rank, dt);
        // if(rank==0)
        //     printf("%d---\n", curRank);
        ncclGroupStart();
        ft::ftNcclSend(m1_ptr+rank*m1_chunkSize, m1_chunkSize, ring_fwd(rank, dt), tensor_para, streams[2]);
        ft::ftNcclRecv(recv_buf+curRank*m1_chunkSize, m1_chunkSize, curRank, tensor_para, streams[curStream]);
        ncclGroupEnd();
        cublasProcessMatMulChunk(streams[curStream], curRank*chunkSize, (curRank+1)*chunkSize, 
                                static_cast<const T2 *>(&alpha),
                                static_cast<const T2 *>(&beta),
                                recv_buf, m2_ptr, m1m2_ptr, M/world_size, N, K, Dtype, computeType);
        curStream ^= 1;
    }

    // cudaEventRecord(events[2], streams[2]);
    // cudaEventRecord(events[0], streams[0]);
    // cudaEventRecord(events[1], streams[1]);

    // for (int i = 0; i < streamNum; i++)
    //     cudaStreamWaitEvent(streams[i], events[i]);
    waitOnStreams(streams, streamNum);
}

template<typename T1, typename T2>
void GemmAllgatherOp::cublasProcessMatMulChunk(cudaStream_t stream,
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

static auto fasterTransformerGemmAllgatherTHS =
#ifdef LEGACY_THS
    torch::jit::class_<torch_ext::GemmAllgatherOp>("FasterTransformerGemmAllgatherOp")
#else
    torch::jit::class_<torch_ext::GemmAllgatherOp>("FasterTransformer", "GemmAllgatherOp")
#endif
        .def(torch::jit::init<int64_t>())
        .def("GemmAllgather", &torch_ext::GemmAllgatherOp::GemmAllgather);
    