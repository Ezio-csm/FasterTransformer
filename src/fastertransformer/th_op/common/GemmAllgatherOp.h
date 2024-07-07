/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION.  All rights reserved.
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

#include "src/fastertransformer/layers/DynamicDecodeLayer.h"
#include "src/fastertransformer/th_op/th_utils.h"
#include "src/fastertransformer/utils/cuda_bf16_wrapper.h"
#include "src/fastertransformer/utils/nccl_utils.h"

namespace ft = fastertransformer;
namespace th = torch;
namespace torch_ext {

#define waitOnStreams(streams, streamNum) \
    for(int i = 0; i < streamNum; i++) { \
        cudaStreamSynchronize(streams[i]); \
    }

#define ring_fwd(id, x) (((id) + (x)) % world_size)
#define ring_bck(id, x) (((id) - (x) + world_size) % world_size)

struct GemmAllgatherOp: th::jit::CustomClassHolder {

    int rank;
    int world_size;

    int chunkNum;
    const int streamNum=3;
    cudaEvent_t* events;
    cudaStream_t* streams;

    ft::NcclParam tensor_para;
    ft::NcclParam pipeline_para;

    at::ScalarType st_;

    GemmAllgatherOp(int64_t _streamNum);
    ~GemmAllgatherOp();

    void GemmAllgather(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2,
                                    int64_t M, int64_t N, int64_t K);

    template<typename T1, typename T2>
    void GemmAllgatherDo(const th::Tensor& m1, const th::Tensor& m2, th::Tensor& m1m2, const T2& alpha, const T2& beta,
                                    int64_t& M, int64_t& N, int64_t& K, cudaDataType_t Dtype, cudaDataType computeType);

    template<typename T1, typename T2>
    void cublasProcessMatMulChunk(cudaStream_t stream,
                              size_t chunkStart, size_t chunkEnd,
                              const T2* alpha, const T2* beta,
                              const T1* m1, const T1* m2, T1* m1m2,
                              int64_t M, int64_t N, int64_t K, cudaDataType_t Dtype, cudaDataType computeType);
};

}  // namespace torch_ext
