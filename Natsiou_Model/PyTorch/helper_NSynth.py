#  This will eventually handle the slimming down of the dataset.
# from torchaudio.transforms import SpectralCentroid
import random

import librosa
import torchaudio
from pathlib import Path
from torchaudio import transforms
from torchaudio import functional
from torchdata.datapipes.iter import IterableWrapper, FileOpener
from torch.utils.data import Dataset
# from audiolazy import lazy_midi
import torch
import matplotlib.pyplot as plt
import os
from IPython.display import Audio


class NSynthDataset(Dataset):

    def __init__(self, annotations_json, audio_dir, transformation, target_sample_rate):
        self.annotations = annotations_json
        self.audio_dir = audio_dir
        self.transformation = transformation
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        signal = self.transformation(signal)
        return signal, label, audio_sample_path

    def _resample_if_necessary(self, signal, sr):
        if sr != self.target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.target_sample_rate)
            signal = resampler(signal)
        return signal

    def _mix_down_if_necessary(self, signal):
        if signal.shape[0] > 1:
            signal = torch.mean(signal, dim=0, keepdim=True)
        return signal

    def _get_audio_sample_path(self, index):
        path = self.audio_dir + "/" + list(self.annotations.items())[index][1]["note_str"] + ".wav"
        return path

    def _get_audio_sample_label(self, index):
        return list(self.annotations.items())[index][1]["note_str"]

    # def _get_fundamental_freq(self, index):
    #     return lazy_midi.midi2freq(list(self.annotations.items())[index][1]["pitch"])

    def get_harmonic_amplitude(self, index, num_harmonics):
        # fund = self._get_fundamental_freq(self, index)
        sample = librosa.load(self._get_audio_sample_path(index))
        hop_length = 512
        n_fft = 2048
        fts = librosa.stft(sample, n_fft=n_fft, hop_length=hop_length)
        S = librosa.amplitude_to_db(abs(fts))
        # for i in range(num_harmonics-1):
        #     harm_freq = fund * (i+1)
        plt.figure(figsize=(15, 5))
        librosa.display.specshow(S, self.sr, hop_length, x_axis="time", y_axis="linear")
        plt.colorbar(format='%+2.0f dB')

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


def plot_waveform(waveform, sr, title="Waveform", ax=None):
    waveform = waveform.numpy()

    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sr

    if ax is None:
        _, ax = plt.subplots(num_channels, 1)
    ax.plot(time_axis, waveform[0], linewidth=1)
    ax.grid(True)
    ax.set_xlim([0, time_axis[-1]])
    ax.set_title(title)


def plot_spectrogram(specgram, title=None, ylabel="freq_bin", ax=None):
    if ax is None:
        _, ax = plt.subplots(1, 1)
    if title is not None:
        ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.imshow(librosa.power_to_db(specgram), origin="lower", aspect="auto", interpolation="nearest")


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


if __name__ == '__main__':
    SAMPLE_RATE = 16000
    json_dict = parse_json_metadata()
    json_dict = filter_json_metadata_pitch(json_dict)
    json_dict = filter_json_metadata_instrument_family(json_dict)
    json_dict = filter_json_metadata_quality(json_dict)

    mel_spectrogram = torchaudio.transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=1024,
        hop_length=512,
        n_mels=64
    )
    data = NSynthDataset(json_dict, "./nsynth-train/audio", mel_spectrogram, SAMPLE_RATE)

    signal, label, path = data[0]
    print(label)

    # Load audio
    SPEECH_WAVEFORM, SAMPLE_RATE = torchaudio.load(path)

    # Define transform
    spectrogram = transforms.Spectrogram(n_fft=512)

    # Perform transform
    spec = spectrogram(SPEECH_WAVEFORM)

    fig, axs = plt.subplots(2, 1)
    plot_waveform(SPEECH_WAVEFORM, SAMPLE_RATE, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram", ax=axs[1])
    fig.tight_layout()
    plt.show()


    # data.get_harmonic_amplitude(0, 7)
    # calculate_spectral_centroid()
    # calculate_fundamental_frequency()
