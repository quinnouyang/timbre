import timbre

from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader

from timbre.data.nsynth import NSynthDataset
from timbre.train.utils import train, test, plot
from timbre.model.vae import VAE


def main() -> None:
    print(timbre.DEVICE)
    print("Loading datasets and dataloaders...")
    # [TODO] Generalize datasets and dataloaders
    # [TODO] Feature extraction
    TRAIN_DATA = NSynthDataset(
        timbre.SOURCES_DIR / "nsynth" / "nsynth-valid" / "examples.json",
        timbre.SOURCES_DIR / "nsynth" / "nsynth-valid" / "audio",
    )
    TEST_DATA = NSynthDataset(
        timbre.SOURCES_DIR / "nsynth" / "nsynth-test" / "examples.json",
        timbre.SOURCES_DIR / "nsynth" / "nsynth-test" / "audio",
    )
    TRAIN_LOADER = DataLoader(
        TRAIN_DATA,
        batch_size=timbre.BATCH_SIZE,
        shuffle=True,
        num_workers=timbre.NUM_WORKERS,
        pin_memory=timbre.USE_PIN_MEMORY,
    )
    TEST_LOADER = DataLoader(
        TEST_DATA,
        batch_size=timbre.BATCH_SIZE,
        shuffle=False,
        num_workers=timbre.NUM_WORKERS,
        pin_memory=timbre.USE_PIN_MEMORY,
    )
    print(
        f"Training datapoints: {len(TRAIN_DATA)}\nTesting datapoints: {len(TEST_DATA)}\n"
    )

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
    # [TODO] Arguments
    main()
