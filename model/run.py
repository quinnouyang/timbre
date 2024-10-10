import torch
import torch.multiprocessing as mp
import run

if __name__ == "__main__":
    n_gpus = torch.cuda.device_count()
    assert n_gpus >= 2, f"Requires at least 2 GPUs to run, but got {n_gpus}"
    world_size = n_gpus // 2

    mp.spawn(run.run, args=(world_size,), nprocs=world_size, join=True)
