import os
import socket
import torch
import torch.distributed as dist


def main():
    info = {
        "hostname": socket.gethostname(),
        "master_addr": os.environ.get("MASTER_ADDR"),
        "master_port": os.environ.get("MASTER_PORT"),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_count": torch.cuda.device_count(),
    }
    print("[dist-env] environment:")
    for k, v in info.items():
        print(f"  {k}: {v}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    print(f"[dist-env] current device: {torch.cuda.current_device()} -> {torch.cuda.get_device_name(torch.cuda.current_device())}")

    if not dist.is_initialized():
        print("[dist-env] torch.distributed not initialized; initializing with world_size=1 for probe")
        dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1, device_id=local_rank)
    print(f"[dist-env] dist backend: {dist.get_backend()}")
    print(f"[dist-env] dist rank/world: {dist.get_rank()}/{dist.get_world_size()}")

    # small collective sanity check
    t = torch.tensor([local_rank], device=torch.cuda.current_device())
    dist.all_reduce(t)
    print(f"[dist-env] all_reduce result: {t.item()}")


if __name__ == "__main__":
    main()
