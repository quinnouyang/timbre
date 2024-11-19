import torch
import torch.distributed as dist

from datetime import datetime
from torch.distributed.elastic.multiprocessing.errors import record
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import datasets
from torchvision.transforms import v2

from reference_models.heidenreich.train import train, test
from reference_models.heidenreich.vae import VAE


@record
def run() -> None:
    learning_rate = 1e-3
    weight_decay = 1e-2
    num_epochs = 50
    latent_dim = 2
    hidden_dim = 512

    batch_size = 128

    transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Lambda(lambda x: x.view(-1) - 0.5),
        ]
    )

    # Download and load the training data
    train_data = datasets.MNIST(
        "~/.pytorch/MNIST_data/",
        download=True,
        train=True,
        transform=transform,
    )
    # Download and load the test data
    test_data = datasets.MNIST(
        "~/.pytorch/MNIST_data/",
        download=True,
        train=False,
        transform=transform,
    )

    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    model = VAE(input_dim=784, hidden_dim=hidden_dim, latent_dim=latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), weight_decay=weight_decay)
    writer = SummaryWriter(f'runs/mnist/vae_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

    dist.init_process_group("nccl")
    rank = dist.get_rank()
    print(f"Start running on rank {rank}.")

    # create model and move it to GPU with id rank
    device_id = rank % torch.cuda.device_count()
    print(f"Device ID: {device_id}")
    model = DDP(model.to(device_id), device_ids=[device_id])

    prev_updates = 0
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        prev_updates = train(
            model, train_loader, optimizer, prev_updates, batch_size, device, writer
        )
        test(model, test_loader, prev_updates, latent_dim, device, writer)

    dist.destroy_process_group()


if __name__ == "__main__":
    run()
