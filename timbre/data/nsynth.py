import json
from argparse import ArgumentParser
from contextlib import suppress
from os import listdir, path
from pathlib import Path
from typing import Any, Literal, NamedTuple

from torch import Tensor, float32
from torch.utils.data import Dataset
from torchaudio import load, transforms

transform_melspec = transforms.MelSpectrogram(n_fft=512, n_mels=64)


def forward_transform(x: Tensor) -> Tensor:
    return transform_melspec(x).flatten().to(float32)


NSynthExample = NamedTuple(
    "NSynthExample",
    [
        ("note", int),
        ("note_str", str),
        ("instrument", int),
        ("instrument_str", str),
        ("pitch", int),
        ("velocity", int),
        ("sample_rate", int),
        ("audio", Tensor),
        ("qualities", list[Literal[0, 1]]),
        # ("qualities_str", list[str]),  # Exclude because it raises a RuntimeError from the default DataLoader collate_fn when it varies in length
        ("instrument_family", int),
        ("instrument_family_str", str),
        ("instrument_source", int),
        ("instrument_source_str", str),
    ],
)
"NSynth example [features](https://magenta.tensorflow.org/datasets/nsynth#example-features)"


class NSynthDataset(Dataset):
    """A subset of the [NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth)

    Parameters
    ----------
    `source_dir` : `str` | `Path`
        Path to a subset of the unprocessed NSynth dataset directory containing `examples.json` and `audio` directory
    """

    def __init__(self, source_dir: str | Path) -> None:
        if isinstance(source_dir, str):
            source_dir = Path(source_dir)

        annotations_file = source_dir / "examples.json"
        audio_dir = source_dir / "audio"

        with open(annotations_file, "r") as f:
            self.annotations: dict[str, Any] = json.load(f)
            self.keys = sorted(self.annotations.keys())  # note_strs
            self.audio_dir = audio_dir
            self.audio_filenames = sorted(
                path.splitext(path.basename(f))[0] for f in listdir(audio_dir)
            )

            # Verify that audio_dir and annotations_file alphabetically map 1:1
            assert (
                self.keys == self.audio_filenames
            ), f"Expected every key/note_str from annotations_file to match every filename from audio_dir\n Instead, audio_dir is missing {set(self.keys) - set(self.audio_filenames)} and/or annotations_file is mising {set(self.audio_filenames) - set(self.keys)}"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i: int) -> Tensor:
        """Load the `i`-th example

        Parameters
        ----------
        `i` : `int`
            Example index (by alphabetical order of the example filename/"note_str")

        Returns
        -------
        Flattened magnitude spectrogram of the example
        # NSynth example features : `NSynthExample`
        #     See the [NSynth example features](https://magenta.tensorflow.org/datasets/nsynth#example-features)
        """
        annotation: dict = self.annotations[self.keys[i]]
        with suppress(KeyError):  # Remove the "qualities_str" key, if not already
            annotation.pop("qualities_str")

        return forward_transform(
            load(path.join(self.audio_dir, self.audio_filenames[i] + ".wav"))[0][0]
        )

        return NSynthExample(
            **annotation,
            audio=from_numpy(
                read(path.join(self.audio_dir, self.audio_filenames[i] + ".wav"))[1]
            )
            / 32768.0,  # 2**15 (to normalize 16-bit samples to [-1, 1])
        )


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("--subset-path", type=str, required=True)

    args = p.parse_args()

    print(f"TEST: Loading NSynthDataset and printing its last example")

    D = NSynthDataset(args.subset_path)

    print(D[-1])
