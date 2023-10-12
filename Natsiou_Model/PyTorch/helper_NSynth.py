#  This will eventually handle the slimming down of the dataset.
# from torchaudio.transforms import SpectralCentroid
import random

import torchaudio
from pathlib import Path
from torchaudio import transforms
from torchaudio import functional
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import Dataset
import os


class NSynthDataset(Dataset):

    def __init__(self, annotations_json, audio_dir):
        self.annotations = annotations_json
        self.audio_dir = audio_dir

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        return signal, label

    def _get_audio_sample_path(self, index):
        path = os.path.join(self.audio_dir, list(self.annotations.items())[index][1]["instrument_str"], ".wav")
        return path

    def _get_audio_sample_label(self, index):
        return list(self.annotations.items())[index][1]["instrument_str"]

    def get_random_annotation(self):
        return random.choice(list(self.annotations.items()))[1]


def get_name(path_and_stream):
    return os.path.basename(path_and_stream[0]), path_and_stream[1]


def parse_json_metadata():
    datapipe1 = IterableWrapper(["nsynth-train/examples.json"])
    datapipe2 = FileOpener(datapipe1, mode="b")
    datapipe3 = datapipe2.map(get_name)
    json_dp = datapipe3.parse_json_files()
    json_list = list(json_dp)
    print("NSynth sample set size:", "->", len(json_list[0][1]))
    return json_list[0][1]


def filter_json_metadata_pitch(old_dict):
    new_dict = {}
    for key in old_dict.keys():
        if old_dict[key]['pitch'] < 40 or old_dict[key]['pitch'] > 96:
            pass
        else:
            new_dict[key] = old_dict[key]
    print("Pitch filtered sample set size: ", "->", len(new_dict))
    return new_dict


def filter_json_metadata_instrument_family(old_dict):
    new_dict = {}
    for key in old_dict.keys():
        if (old_dict[key]['instrument_family_str'] == 'bass' or
                old_dict[key]['instrument_family_str'] == 'brass' or
                old_dict[key]['instrument_family_str'] == 'flute' or
                old_dict[key]['instrument_family_str'] == 'guitar' or
                old_dict[key]['instrument_family_str'] == 'keyboard' or
                old_dict[key]['instrument_family_str'] == 'organ' or
                old_dict[key]['instrument_family_str'] == 'reed' or
                old_dict[key]['instrument_family_str'] == 'string' or
                old_dict[key]['instrument_family_str'] == 'vocal'):
            new_dict[key] = old_dict[key]
    print("Instrument filtered sample set size: ", "->", len(new_dict))
    return new_dict


def filter_json_metadata_quality(old_dict):
    new_dict = {}
    for key in old_dict.keys():
        if ("percussive" in old_dict[key]['qualities_str'] or
                "fast_decay" in old_dict[key]['qualities_str'] or
                "long_release" in old_dict[key]['qualities_str'] or
                "multiphonic" in old_dict[key]['qualities_str'] or
                "tempo-synced" in old_dict[key]['qualities_str'] or
                "nonlinear_env" in old_dict[key]['qualities_str']):
            pass
        else:
            new_dict[key] = old_dict[key]
    print("Quality filtered sample set size: ", "->", len(new_dict))
    return new_dict


def calculate_spectral_centroid():
    waveform, sample_rate = torchaudio.load(Path('nsynth-train/audio/guitar_acoustic_001-082-050.wav').resolve(),
                                            normalize=True)
    transform = transforms.SpectralCentroid(sample_rate, n_fft=700,
                                            win_length=690, hop_length=172)
    spectral_centroid = transform(waveform)
    print(spectral_centroid)


def calculate_fundamental_frequency():
    waveform, sample_rate = torchaudio.load(Path('nsynth-train/audio/guitar_acoustic_001-082-050.wav').resolve(),
                                            normalize=True)
    pitch = functional.detect_pitch_frequency(waveform, sample_rate)
    print(pitch)


if __name__ == '__main__':
    json_dict = parse_json_metadata()
    json_dict = filter_json_metadata_pitch(json_dict)
    json_dict = filter_json_metadata_instrument_family(json_dict)
    json_dict = filter_json_metadata_quality(json_dict)
    data = NSynthDataset(json_dict, "/nsynth-train/audio")
    point = data.get_random_annotation()
    print(point)
    # calculate_spectral_centroid()
    # calculate_fundamental_frequency()
