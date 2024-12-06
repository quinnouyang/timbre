import timbre

from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from timbre.data.nsynth import NSynthDataset
from timbre.train.utils import train, test, plot
from timbre.model.vae import VAE


# [TODO] Generalize datasets and dataloaders
# [TODO] Feature extraction (actually figure out I/O with reasonable dimensions)
# [TODO] Get it to run on malleus
# [TODO] Tensorboard
# [TODO] Achieve basic clustering results and audio output
# [TODO] Checkpoints
# [TODO] DDP
# [TODO] Arguments


def build_datasets() -> tuple[NSynthDataset, NSynthDataset]:
    return (
        NSynthDataset(timbre.SOURCES_DIR / "nsynth" / "nsynth-valid"),
        NSynthDataset(timbre.SOURCES_DIR / "nsynth" / "nsynth-test"),
    )


def build_dataloaders() -> tuple[DataLoader, DataLoader]:
    print("Loading datasets and dataloaders...")
    train_data, test_data = build_datasets()
    print(
        f"Training datapoints: {len(train_data)}\nTesting datapoints: {len(test_data)}\n"
    )

    return DataLoader(
        train_data,
        batch_size=timbre.BATCH_SIZE,
        shuffle=True,
        num_workers=timbre.NUM_WORKERS,
        pin_memory=timbre.USE_PIN_MEMORY,
    ), DataLoader(
        test_data,
        batch_size=timbre.BATCH_SIZE,
        shuffle=False,
        num_workers=timbre.NUM_WORKERS,
        pin_memory=timbre.USE_PIN_MEMORY,
    )


def main() -> None:
    TRAIN_LOADER, TEST_LOADER = build_dataloaders()

    print("Initiating model, optimizer, and Tensorboard...")
    MODEL = VAE(timbre.INPUT_DIM, timbre.HIDDEN_DIM, timbre.LATENT_DIM).to(
        timbre.DEVICE
    )
    OPT = AdamW(MODEL.parameters(), weight_decay=timbre.WEIGHT_DECAY)

    print("Entering train-test loop...\n")
    prev_updates = 0
    for epoch in range(timbre.NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{timbre.NUM_EPOCHS}")
        prev_updates = train(
            MODEL,
            TRAIN_LOADER,
            OPT,
            prev_updates,
            timbre.DEVICE,
            timbre.BATCH_SIZE,
            timbre.WRITER,
        )
        test(
            MODEL,
            TEST_LOADER,
            prev_updates,
            timbre.DEVICE,
            timbre.LATENT_DIM,
            timbre.WRITER,
        )

    print("\nPlotting...")
    plot(
        MODEL,
        TRAIN_LOADER,
        timbre.DEVICE,
        timbre.LATENT_DIM,
        timbre.RUNS_DIR,
        timbre.DATETIME_NOW,
    )

    print("Done.")


if __name__ == "__main__":
    main()
