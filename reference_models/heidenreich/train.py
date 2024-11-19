import matplotlib.pyplot as plt
import numpy as np
import torch

from matplotlib.colors import LogNorm
from pathlib import Path
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from .vae import VAE


def train(
    model: VAE | DDP,
    dataloader: DataLoader,
    optimizer: Optimizer,
    prev_updates: int,
    device: torch.device,
    batch_size: int,
    writer: SummaryWriter | None = None,
) -> int:
    """
    Parameters
    ----------
    `model` : `VAE | DDP`
        Single or multi/distributed GPU model
    `dataloader` : `DataLoader`
    `optimizer` : `Optimizer`
    `prev_updates`: `int`
        Number of updates from previous epochs
    `device`: `torch.device`
    `batch_size`: `int`
    `writer`: `SummaryWriter | None = None`

    Returns
    -------
    New number of updates (includes `prev_updates`)
    """
    model.train()
    n_upd = prev_updates

    def try_calculate_grad(loss, output) -> None:
        """
        Calculates, logs, and writes gradients every 100 updates
        """
        if n_upd % 100 != 0:
            return

        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1.0 / 2)

        print(
            f"Step {n_upd:,} (N samples: {n_upd*batch_size:,}), Loss: {loss.item():.4f} (Recon: {output.loss_recon.item():.4f}, KL: {output.loss_kl.item():.4f}) Grad: {total_norm:.4f}"
        )

        if writer is not None:
            global_step = n_upd
            writer.add_scalar("Loss/Train", loss.item(), global_step)
            writer.add_scalar("Loss/Train/BCE", output.loss_recon.item(), global_step)
            writer.add_scalar("Loss/Train/KLD", output.loss_kl.item(), global_step)
            writer.add_scalar("GradNorm/Train", total_norm, global_step)

    def update(data: torch.Tensor) -> None:
        """
        A single update step
        """
        optimizer.zero_grad()
        for param in model.parameters():
            param.grad = None

        output = model(data)
        loss = output.loss

        loss.backward()

        try_calculate_grad(loss, output)

        clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

    for data, _ in tqdm(dataloader):
        n_upd += 1
        update(data.to(device))

    return prev_updates + len(dataloader)


def test(
    model: VAE | DDP,
    dataloader: DataLoader,
    cur_step: int,
    device: torch.device,
    latent_dim: int,
    writer: SummaryWriter | None = None,
) -> None:
    """
    Parameters
    ----------
    `model` : `VAE | DDP`
        Single or multi/distributed GPU model
    `dataloader` : `DataLoader`
    `cur_step`: `int`
        Current update count from previous epochs
    `device`: `torch.device`
    `latent_dim`: `int`
    `writer`: `SummaryWriter | None = None`
    """
    model.eval()
    test_loss = 0
    test_recon_loss = 0
    test_kl_loss = 0

    with torch.no_grad():
        for data, _ in tqdm(dataloader, desc="Testing"):
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten the data

            output = model(data, compute_loss=True)  # Forward pass

            test_loss += output.loss.item()
            test_recon_loss += output.loss_recon.item()
            test_kl_loss += output.loss_kl.item()

    test_loss /= len(dataloader)
    test_recon_loss /= len(dataloader)
    test_kl_loss /= len(dataloader)
    print(
        f"====> Test set loss: {test_loss:.4f} (BCE: {test_recon_loss:.4f}, KLD: {test_kl_loss:.4f})"
    )

    if writer is not None:
        writer.add_scalar("Loss/Test", test_loss, global_step=cur_step)
        writer.add_scalar(
            "Loss/Test/BCE", output.loss_recon.item(), global_step=cur_step
        )
        writer.add_scalar("Loss/Test/KLD", output.loss_kl.item(), global_step=cur_step)

        # Log reconstructions
        writer.add_images(
            "Test/Reconstructions",
            output.x_recon.view(-1, 1, 28, 28),
            global_step=cur_step,
        )
        writer.add_images(
            "Test/Originals", data.view(-1, 1, 28, 28), global_step=cur_step
        )

        # Log random samples from the latent space
        z = torch.randn(16, latent_dim).to(device)
        samples = model.decode(z)
        writer.add_images(
            "Test/Samples", samples.view(-1, 1, 28, 28), global_step=cur_step
        )


def plot(
    model: VAE | DDP,
    train_loader: DataLoader,
    device: torch.device,
    latent_dim: int,
    runs_dir: Path,
    datetime_now: str,
) -> None:
    """
    Parameters
    ----------
    `model` : `VAE | DDP`
        Single or multi/distributed GPU model
    `train_loader` : `DataLoader`
    `device`: `torch.device`
    `latent_dim`: `int`
    `runs_dir`: `Path`
        Path to runs to create a directory to export this run's plots to
    `datetime_now`: `str`
        Datetime to label this run with as a plots directory name
    """
    plot_dir = runs_dir / f"plots_{datetime_now}"
    plot_dir.mkdir(parents=True, exist_ok=True)

    z = torch.randn(64, latent_dim).to(device)
    samples = model.decode(z)
    # samples = torch.sigmoid(samples)

    # print first sample
    # print(samples[0])

    # Plot the generated images
    fig, ax = plt.subplots(8, 8, figsize=(8, 8))
    for i in range(8):
        for j in range(8):
            ax[i, j].imshow(
                samples[i * 8 + j].view(28, 28).cpu().detach().numpy(), cmap="gray"
            )
            ax[i, j].axis("off")

    # plt.show()
    plt.savefig(plot_dir / "vae_mnist.webp")

    # encode and plot the z values for the train set
    model.eval()
    z_all = []
    y_all = []
    with torch.no_grad():
        for data, target in tqdm(train_loader, desc="Encoding"):
            data = data.to(device)
            output = model(data, compute_loss=False)
            z_all.append(output.z_sample.cpu().numpy())
            y_all.append(target.numpy())

    z_all = np.concatenate(z_all, axis=0)
    y_all = np.concatenate(y_all, axis=0)
    plt.figure(figsize=(10, 10))
    plt.scatter(z_all[:, 0], z_all[:, 1], c=y_all, cmap="tab10")
    plt.colorbar()
    # plt.show()
    plt.savefig(plot_dir / "vae_mnist_2d_scatter.webp")

    # plot as 2d histogram, log scale
    plt.figure(figsize=(10, 10))
    plt.hist2d(z_all[:, 0], z_all[:, 1], bins=128, cmap="Blues", norm=LogNorm())
    plt.colorbar()
    # plt.show()
    plt.savefig(plot_dir / "vae_mnist_2d_hist.webp")

    # plot 1d histograms
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].hist(z_all[:, 0], bins=100, color="b", alpha=0.7)
    ax[0].set_title("z1")
    ax[1].hist(z_all[:, 1], bins=100, color="b", alpha=0.7)
    ax[1].set_title("z2")
    # plt.show()
    plt.savefig(plot_dir / "vae_mnist_1d_hist.webp")

    n = 15
    z1 = torch.linspace(-0, 1, n)
    z2 = torch.zeros_like(z1) + 2
    z = torch.stack([z1, z2], dim=-1).to(device)
    samples = model.decode(z)
    samples = torch.sigmoid(samples)

    # Plot the generated images
    fig, ax = plt.subplots(1, n, figsize=(n, 1))
    for i in range(n):
        ax[i].imshow(samples[i].view(28, 28).cpu().detach().numpy(), cmap="gray")
        ax[i].axis("off")

    plt.savefig(plot_dir / "vae_mnist_interp.webp")
