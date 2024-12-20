from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

from reference_models.heidenreich.single.config import (
    DATASETS_DIR,
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
    RUNS_DIR,
    DATETIME_NOW,
)
from reference_models.heidenreich import train, test, plot, VAE


if __name__ == "__main__":
    print("Loading datasets and dataloaders...")
    TRAIN_DATA = MNIST(
        DATASETS_DIR,
        download=True,
        train=True,
        transform=TRANSFORM,
    )
    TEST_DATA = MNIST(
        DATASETS_DIR,
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
    print(
        f"Training datapoints: {len(TRAIN_DATA)}\nTesting datapoints: {len(TEST_DATA)}\n"
    )

    print("Initiating model, optimizer, and Tensorboard...")
    MODEL = VAE(INPUT_DIM, HIDDEN_DIM, LATENT_DIM).to(DEVICE)
    OPT = AdamW(MODEL.parameters(), weight_decay=WEIGHT_DECAY)

    print("Entering train-test loop...\n")
    prev_updates = 0
    for epoch in range(N_EPOCHS):
        print(f"Epoch {epoch+1}/{N_EPOCHS}")
        prev_updates = train(
            MODEL, TRAIN_LOADER, OPT, prev_updates, DEVICE, BATCH_SIZE, WRITER
        )
        test(MODEL, TEST_LOADER, prev_updates, DEVICE, LATENT_DIM, WRITER)

    print("\nPlotting...")
    plot(MODEL, TRAIN_LOADER, DEVICE, LATENT_DIM, RUNS_DIR, DATETIME_NOW)

    print("Done.")
