import json
import torchaudio.transforms as tat

from os import path, listdir
from argparse import ArgumentParser
from pathlib import Path
from typing import NamedTuple, Literal, Any
from torch import Tensor, from_numpy, float32
from torch.utils.data import Dataset
from contextlib import suppress
from model.utils import read

SR = 16000
N_FFT = 2048
HOP_LEN = N_FFT // 4


stft_fn = tat.Spectrogram(N_FFT, hop_length=HOP_LEN, normalized=True)
gla_fn = tat.GriffinLim(N_FFT, hop_length=HOP_LEN)


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
    `annotations_file` : `str` | `Path`
        Path to the JSON file containing the annotations
    `audio_dir` : `str` | `Path`
        Path to the directory containing the audio files
    """

    def __init__(
        self,
        annotations_file: str | Path,
        audio_dir: str | Path,
    ) -> None:
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

        return (
            stft_fn(
                from_numpy(
                    read(path.join(self.audio_dir, self.audio_filenames[i] + ".wav"))[1]
                )
            )
            .flatten()
            .to(float32)
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

    D = NSynthDataset(
        path.join(args.subset_path, "examples.json"),
        path.join(args.subset_path, "audio"),
    )

    print(D[-1])
