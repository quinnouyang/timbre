import json

from os import path, listdir
from argparse import ArgumentParser
from pathlib import Path
from typing import Literal, Any
from torch import Tensor, from_numpy
from torch.utils.data import Dataset
from scipy.io.wavfile import read


class NSynthExample:
    """NSynth example audio and [features](https://magenta.tensorflow.org/datasets/nsynth#example-features)"""

    def __init__(
        self,
        note: int,
        note_str: str,
        instrument: int,
        instrument_str: str,
        pitch: int,
        velocity: int,
        sample_rate: int,
        audio: Tensor,
        qualities: list[Literal[0, 1]],
        qualities_str: list[str],
        instrument_family: int,
        instrument_family_str: str,
        instrument_source: int,
        instrument_source_str: str,
    ) -> None:
        self.note = note
        self.note_str = note_str
        self.instrument = instrument
        self.instrument_str = instrument_str
        self.pitch = pitch
        self.velocity = velocity
        self.sample_rate = sample_rate
        self.audio = audio
        self.qualities = qualities
        self.qualities_str = qualities_str
        self.instrument_family = instrument_family
        self.instrument_family_str = instrument_family_str
        self.instrument_source = instrument_source
        self.instrument_source_str = instrument_source_str

    def __repr__(self) -> str:
        """Represent the example. See the [NSynth example features](https://magenta.tensorflow.org/datasets/nsynth#example-features).

        Returns
        -------
        Primary non-audio features of the example : `str`
            "note" : `int`
                A unique integer identifier for the note
            "note_str" : `str`
                A unique string identifier for the note in the format `<instrument_str>-<pitch>-<velocity>`
            "pitch" : `int`
                The 0-based MIDI pitch in the range `[0, 127]`
            "qualities_str" : `int`
                A list IDs of which qualities are present in this note selected from the sonic qualities list
        """
        return f"NSynthExample(note={self.note}, note_str={self.note_str}, pitch={self.pitch}, qualities_str={self.qualities_str})"


class NSynthDataset(Dataset):
    """[NSynth dataset](https://magenta.tensorflow.org/datasets/nsynth)

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
                len(self.keys) == len(self.audio_filenames)
                and self.keys == self.audio_filenames
            ), f"Expected every key/note_str in annotations_file to match every filename in audio_dir.\n{set(self.keys) - set(self.audio_filenames)} is missing in audio_dir and/or {set(self.audio_filenames) - set(self.keys)} is missing in annotations_file"

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, i: int) -> tuple[Tensor, int, str, int, list[str]]:
        """Load the `i`-th example from the dataset. See the [NSynth example features](https://magenta.tensorflow.org/datasets/nsynth#example-features).

        Parameters
        ----------
        `i` : `int`
            Example index (by alphabetical order of the example filename/"note_str")

        Returns
        -------
        Audio and primary features of the example : `tuple`
            "audio" : `Tensor`
                A list of audio samples represented as floating point values in the range `[-1, 1]`
            "note" : `int`
                A unique integer identifier for the note
            "note_str" : `str`
                A unique string identifier for the note in the format `<instrument_str>-<pitch>-<velocity>`
            "pitch" : `int`
                The 0-based MIDI pitch in the range `[0, 127]`
            "qualities_str" : `int`
                A list IDs of which qualities are present in this note selected from the sonic qualities list
        """
        e = NSynthExample(
            audio=from_numpy(
                read(path.join(self.audio_dir, self.audio_filenames[i] + ".wav"))[1]
            )
            / 32768.0,  # 2**15 (to normalize 16-bit samples to [-1, 1])
            **self.annotations[self.keys[i]],
        )
        return e.audio, e.note, e.note_str, e.pitch, e.qualities_str


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
