# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 MASTER_ADDR=127.0.0.1 MASTER_PORT=1234 mpirun -n 8 --allow-run-as-root python gemmAllreduce_test.py
# pip3 install transformers -i https://pypi.tuna.tsinghua.edu.cn/simple

import os

import torch
import torch.distributed
from contextlib import contextmanager
from typing import Optional

import torch
import torch.distributed as dist

# from logging import logger

import time

# logger = logger.getLogger(__name__)
# torch.classes.load_library("./lib/libth_transformer.so")
# torch.classes.load_library("/usr/src/llm-inference/FasterTransformer/build/lib/libth_transformer.so")
# torch.classes.load_library(
#     "/opt/conda/lib/python3.10/site-packages/fastertransformer/libth_transformer.so"
# )
torch.classes.load_library(
    "/home/ubuntu/csm/FasterTransformer/build/lib/libth_transformer.so"
)

rank = int(
    os.environ.get("OMPI_COMM_WORLD_RANK", default=os.environ.get("LOCAL_RANK", -1))
)  # LOCAL_RANK
world_size = int(
    os.environ.get("OMPI_COMM_WORLD_SIZE", default=os.environ.get("WORLD_SIZE", -1))
)
# print(f'world_size: {world_size}, local_rank: {rank}')
torch.distributed.init_process_group(
    backend="nccl",  # 通信后端
    init_method="env://",  # 初始化方法
    world_size=world_size,  # 设备总数
    rank=rank,  # 当前设备的局部rank
)

process_group = torch.distributed.group.WORLD
print(
    f"process_group:{process_group}, rank:{process_group.rank()}, world_size:{process_group.size()}"
)
torch.cuda.set_device(rank)

torch.cuda.manual_seed(123)
torch.manual_seed(123)

_SUPPORTED_WORLD_SIZES = [2, 4, 6, 8]
try:
    import pynvml
except ImportError:
    pynvml = None


@contextmanager
def _nvml():
    try:
        pynvml.nvmlInit()
        yield
    finally:
        pynvml.nvmlShutdown()


# query if the set of gpus are fully connected by nvlink (1 hop)
@_nvml()
def _is_full_nvlink(rank, world_size):
    handle = pynvml.nvmlDeviceGetHandleByIndex(rank)
    for i in range(world_size):
        if i != rank:
            try:
                link_state = pynvml.nvmlDeviceGetNvLinkState(handle, i)
                if not link_state:
                    return False
            except pynvml.NVMLError as error:
                print(
                    f'NVLink detection failed with message "{str(error)}". '
                    "This is normal if your machine has no NVLink equipped"
                )
                return False
    return True


def _can_p2p(rank: int, world_size: int) -> bool:
    for i in range(world_size):
        if i == rank:
            continue
        if not torch.cuda.can_device_access_peer(rank, i):
            return False
    return True


if world_size not in _SUPPORTED_WORLD_SIZES:
    print(
        "Custom allreduce is disabled due to an unsupported world size: "
        "%d. Supported world sizes: %s. To silence this warning, specify"
        "disable_custom_all_reduce=True explicitly.",
        world_size,
        str(_SUPPORTED_WORLD_SIZES),
    )
if not _can_p2p(rank, world_size):
    print(
        "Custom allreduce is disabled because your platform lacks GPU P2P"
        " capability. To silence this warning, specify"
        "disable_custom_all_reduce=True explicitly."
    )

is_full_nvlink = _is_full_nvlink(rank, world_size)

# ca_handle = torch.classes.FasterTransformer.CustomAllReduceWrapper(
#     world_size, 4 * 1024 * 1024, True, torch.bfloat16, True
# )

@torch.inference_mode()
def test_gemmAllreduce(bs_seq_len, hidden_size, intermediate_size, dtype, my_handle, warp_up=5, iter_cnt=200, nStream=2):
    g_process_group = torch.distributed.group.WORLD

    single_intermediate_size = intermediate_size // world_size

    data = torch.randn(bs_seq_len, single_intermediate_size, dtype=dtype, device=f'cuda:{rank}')
    weight = torch.randn(single_intermediate_size, hidden_size, dtype=dtype, device=f'cuda:{rank}')
    # data = torch.tensor(data, dtype=torch.float16).reshape(bs_seq_len, single_intermediate_size).to(f'cuda:{rank}')
    # weight = torch.tensor(weight, dtype=torch.float16).reshape(single_intermediate_size, hidden_size).to(f'cuda:{rank}')
    # if rank == 0:
    #     print(f'rank: {rank},\n data: {data},\n weight: {weight}')

    torch.cuda.synchronize()
    for _ in range(warp_up):
        out = torch.matmul(data, weight)
        if world_size > 1:
            torch.distributed.all_reduce(out, group=g_process_group)
    torch.cuda.synchronize()

    ################################################################
    # start_time = time.time()
    # for _ in range(iter_cnt):
    #     out = torch.matmul(data, weight)
    #     if world_size > 1:
    #         torch.distributed.all_reduce(out, group=g_process_group)
    # torch.cuda.synchronize()
    # end_time = time.time()
    # time1 = (end_time - start_time) / iter_cnt
    # if rank == 0:
    #     print(f'Reference cost time: {(end_time - start_time)/iter_cnt}')

    ################################################################
    out_2 = torch.zeros_like(out)
    
    start_time = time.time()
    for _ in range(iter_cnt):
        my_handle.GemmAllreduce(data, weight, out_2, bs_seq_len, hidden_size, single_intermediate_size)
    torch.cuda.synchronize()
    end_time = time.time()
    time2 = (end_time - start_time) / iter_cnt

    ################################################################

    
    # split_hidden_size = hidden_size // nStream
    # steams = [torch.cuda.Stream() for _ in range(nStream)]

    # start_time = time.time()
    # for _ in range(iter_cnt):
    #     out3 = []
    #     for i in range(0, nStream):
    #         stream_0 = steams[i]
    #         with torch.cuda.stream(stream_0):
    #             tmp = torch.matmul(data, weight[:,i*split_hidden_size:(i+1)*split_hidden_size])
    #             if g_process_group.size() > 1:
    #                 torch.distributed.all_reduce(tmp , group=g_process_group)
    #             out3.append(tmp)

    # torch.cuda.synchronize()
    # for stream in steams:
    #     stream.synchronize()
    # end_time = time.time()
    # time3 = (end_time - start_time) / iter_cnt

    # if rank == 0: 
    #     print(f'Split-Reduce2 cost time: {(end_time - start_time)/iter_cnt}')

    # if rank == 0:
    #     print(f'out:{out}, \nsum:{out.sum()}')
    #     print(f'out2: {out_2}, \nsum:{out_2.sum()}')

    return time2

import pandas as pd

bs_seq_len_list_base   = [1, 2, 4, 8, 16, 32, 64, 128]
# bs_seq_len_list_base   = [1, 2, 4, 8, 16]
bs_seq_len_list        = [t*1024 for t in bs_seq_len_list_base]
hidden_size_list       = [6656, 8192, 12288]
intermediate_size_lsit = [6656, 8192, 12288, 32768]
# intermediate_size_lsit = [6656, 8192, 12288]
# dtype_list             = [torch.float16, torch.bfloat16, torch.float32]
dtype_list             = [torch.float16]
if __name__ == "__main__":
    nChunk = 16
    my_handle = torch.classes.FasterTransformer.GemmAllreduceOp(nChunk)

    if rank==0:
        df = pd.DataFrame(columns=['bs_seq_len', 'hidden_size', 'intermediate_size', 'dtype', 'time_myfunc'])

    for dtype in dtype_list:
        for bs_seq_len in bs_seq_len_list:
            for hidden_size in hidden_size_list:
                for intermediate_size in intermediate_size_lsit:
                    if hidden_size != intermediate_size and (not (hidden_size == 12288 and intermediate_size == 32768)):
                        continue
                    if dtype == torch.float32 and hidden_size == 12288 and intermediate_size == 32768:
                        continue
                    time2 = test_gemmAllreduce(bs_seq_len, hidden_size, intermediate_size, dtype, my_handle, nStream=2)
                    if rank==0:
                        log_info = f'bs_seq_len: {bs_seq_len}, hidden_size: {hidden_size}, intermediate_size: {intermediate_size}, dtype: {dtype}, time_myfunc: {time2}'
                        print(log_info)
                        new_row = pd.DataFrame([[bs_seq_len, hidden_size, intermediate_size, dtype, time2]], columns=['bs_seq_len', 'hidden_size', 'intermediate_size', 'dtype', 'time_myfunc'])
                        df = pd.concat([df, new_row], ignore_index=True)
    if rank==0:
        df.to_csv(f'test_{world_size}_16.csv', index=False)
