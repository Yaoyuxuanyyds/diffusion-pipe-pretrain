import argparse
import os
import time
from datetime import timedelta
from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.distributed as dist
import deepspeed


def log(msg, rank):
    print(f"[rank {rank}] {msg}", flush=True)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(self, x):
        return self.net(x).mean()


def init_torch_dist(local_rank, backend, timeout=300):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if dist.is_initialized():
        return rank, world_size

    dist.init_process_group(backend=backend, init_method="env://", timeout=timedelta(seconds=timeout))
    return rank, world_size


def allreduce_sanity(local_rank, rank, world_size, backend):
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    tensor = torch.tensor([float(rank)], device=device)
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    expected = sum(range(world_size))
    log(f"{backend} all_reduce result={tensor.item()} expected={expected}", rank)
    if abs(tensor.item() - expected) > 1e-3:
        raise RuntimeError("All-reduce mismatch; check device mapping / networking.")


def deepspeed_sanity(local_rank, rank, args):
    model = ToyModel()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    model.to(device)

    ds_config = {
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "zero_optimization": {"stage": 0},
        "optimizer": {
            "type": "Adam",
            "params": {
                "lr": 1e-3,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
            },
        },
        "steps_per_print": 1,
    }
    ds_args = SimpleNamespace(local_rank=local_rank)
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=ds_args,
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
    )
    dummy = torch.randn(2, 4, device=device)
    loss = model_engine(dummy)
    model_engine.backward(loss)
    model_engine.step()
    log(f"DeepSpeed forward/backward ok, loss={loss.item():.4f}", rank)


def main():
    parser = argparse.ArgumentParser(description="Minimal distributed diagnostics for NCCL/DeepSpeed.")
    parser.add_argument("--skip-deepspeed", action="store_true", help="Only run torch.distributed checks.")
    parser.add_argument("--timeout", type=int, default=300, help="Process group timeout seconds.")
    # Accepted for compatibility with deepspeed/torchrun launchers.
    parser.add_argument("--local_rank", type=int, default=None, help="Local rank passed by the launcher.")
    args = parser.parse_args()

    local_rank = args.local_rank if args.local_rank is not None else int(os.environ.get("LOCAL_RANK", 0))
    backend = "nccl" if torch.cuda.is_available() else "gloo"

    torch.cuda.set_device(local_rank) if torch.cuda.is_available() else None

    rank, world_size = init_torch_dist(local_rank, backend, timeout=args.timeout)
    log(f"backend={backend}, world_size={world_size}, local_rank={local_rank}, cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES')}", rank)
    log(f"torch={torch.__version__}, deepspeed={deepspeed.__version__}, cuda_available={torch.cuda.is_available()}", rank)

    start = time.time()
    allreduce_sanity(local_rank, rank, world_size, backend)
    dist.barrier()
    log(f"all_reduce + barrier completed in {time.time() - start:.2f}s", rank)

    if not args.skip_deepspeed:
        deepspeed_sanity(local_rank, rank, args)
        dist.barrier()
        log("DeepSpeed barrier after step completed", rank)

    dist.destroy_process_group()
    log("Finished diagnostics", rank)


if __name__ == "__main__":
    main()