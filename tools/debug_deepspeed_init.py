"""Run with torchrun/deepspeed to verify device mapping at init time.
Example (single GPU):
  torchrun --nproc_per_node=1 tools/debug_deepspeed_init.py
"""
import os
import torch
import torch.distributed as dist
import deepspeed


def log(msg):
    rank = int(os.environ.get("RANK", 0))
    print(f"[rank {rank}] {msg}", flush=True)


def main():
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    log(f"set torch device -> {torch.cuda.current_device()} ({torch.cuda.get_device_name(torch.cuda.current_device())})")

    # Explicitly init process group with device_id to avoid NCCL unknown device mapping
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            rank=rank,
            world_size=world_size,
            device_id=local_rank,
        )
    log(f"dist backend={dist.get_backend()} rank/world={dist.get_rank()}/{dist.get_world_size()}")

    # probe a small collective to ensure mapping works
    t = torch.tensor([local_rank], device=torch.cuda.current_device())
    dist.all_reduce(t)
    log(f"all_reduce result {t.item()}")

    # initialize DeepSpeed; should reuse existing pg
    deepspeed.init_distributed(dist_backend="nccl", init_method="env://")
    log("DeepSpeed init completed")


if __name__ == "__main__":
    main()
