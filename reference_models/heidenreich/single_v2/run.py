from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from reference_models.heidenreich.single_v2.config import (
    DATA_DIR,
    TRANSFORM,
    BATCH_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    INPUT_DIM,
    HIDDEN_DIM,
    LATENT_DIM,
    DEVICE,
    WEIGHT_DECAY,
    WRITER,
    N_EPOCHS,
)
from reference_models.heidenreich.single_v2.train import train, test, plot
from reference_models.heidenreich.vae import VAE


if __name__ == "__main__":
    print("Loading datasets and dataloaders")
    TRAIN_DATA = MNIST(
        DATA_DIR,
        download=True,
        train=True,
        transform=TRANSFORM,
    )
    TEST_DATA = MNIST(
        DATA_DIR,
        download=True,
        train=False,
        transform=TRANSFORM,
    )
    TRAIN_LOADER = DataLoader(
        TRAIN_DATA,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )
    TEST_LOADER = DataLoader(
        TEST_DATA,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
    )

    print("Initiating model, optimizer, and Tensorboard")
    MODEL = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    OPT = AdamW(MODEL.parameters(), weight_decay=WEIGHT_DECAY)

    print("Entering train-test loop...\n")
    prev_updates = 0
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        prev_updates = train(MODEL, TRAIN_LOADER, OPT, prev_updates, writer=WRITER)
        test(MODEL, TEST_LOADER, prev_updates, writer=WRITER)

    print("Plotting...")
    plot(MODEL, TRAIN_LOADER)

    print("Done.")
