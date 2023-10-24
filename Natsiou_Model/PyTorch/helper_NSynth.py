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
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
from IPython.display import Audio


class NSynthDataset(Dataset):
    def __init__(self, annotations_json, audio_dir, target_sample_rate):
        self.annotations = annotations_json
        self.audio_dir = audio_dir
        self.target_sample_rate = target_sample_rate

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        audio_sample_path = self._get_audio_sample_path(index)
        label = self._get_audio_sample_label(index)
        signal, sr = torchaudio.load(audio_sample_path)
        signal = self._resample_if_necessary(signal, sr)
        signal = self._mix_down_if_necessary(signal)
        midi = self._get_audio_sample_midi(index)
        fund_freq = self._get_fundamental_freq(midi)
        attack_time = self._get_attack_time_from_waveform(signal)
        return signal, label, audio_sample_path, fund_freq, attack_time

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
        path = (
            self.audio_dir
            + "/"
            + list(self.annotations.items())[index][1]["note_str"]
            + ".wav"
        )
        return path

    def _get_audio_sample_label(self, index):
        return list(self.annotations.items())[index][1]["note_str"]

    def _get_audio_sample_midi(self, index):
        return list(self.annotations.items())[index][1]["pitch"]

    def _get_fundamental_freq(self, midi):
        return 440 * 2 ** ((midi - 69) / 12)

    def _get_attack_time_from_waveform(self, signal):
        # First, calculate max amplitude of the waveform.
        w = signal.numpy()[0]
        max_amp = max(w[0:SAMPLE_RATE])
        max_amp_idx = list(w).index(max_amp)
        # Then, find when the waveform first gets to 10 percent of that amplitude
        for idx, i in enumerate(w):
            if i > max_amp / 10:
                ten_percent_idx = idx
                break
        # Then, do the same thing but for 90 percent
        for idx, i in enumerate(w):
            if i > 9 * max_amp / 10:
                ninety_percent_idx = idx
                break
        attack_time = (ninety_percent_idx - ten_percent_idx) / SAMPLE_RATE
        return attack_time

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
        if old_dict[key]["pitch"] < 40 or old_dict[key]["pitch"] > 96:
            pass
        else:
            new_dict[key] = old_dict[key]
    print("Pitch filtered sample set size: ", "->", len(new_dict))
    return new_dict


def filter_json_metadata_instrument_family(old_dict):
    new_dict = {}
    for key in old_dict.keys():
        if (
            old_dict[key]["instrument_family_str"] == "bass"
            or old_dict[key]["instrument_family_str"] == "brass"
            or old_dict[key]["instrument_family_str"] == "flute"
            or old_dict[key]["instrument_family_str"] == "guitar"
            or old_dict[key]["instrument_family_str"] == "keyboard"
            or old_dict[key]["instrument_family_str"] == "organ"
            or old_dict[key]["instrument_family_str"] == "reed"
            or old_dict[key]["instrument_family_str"] == "string"
            or old_dict[key]["instrument_family_str"] == "vocal"
        ):
            new_dict[key] = old_dict[key]
    print("Instrument filtered sample set size: ", "->", len(new_dict))
    return new_dict


def filter_json_metadata_quality(old_dict):
    new_dict = {}
    for key in old_dict.keys():
        if (
            "percussive" in old_dict[key]["qualities_str"]
            or "fast_decay" in old_dict[key]["qualities_str"]
            or "long_release" in old_dict[key]["qualities_str"]
            or "multiphonic" in old_dict[key]["qualities_str"]
            or "tempo-synced" in old_dict[key]["qualities_str"]
            or "nonlinear_env" in old_dict[key]["qualities_str"]
        ):
            pass
        else:
            new_dict[key] = old_dict[key]
    print("Quality filtered sample set size: ", "->", len(new_dict))
    return new_dict


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
    ax.imshow(
        librosa.power_to_db(specgram),
        origin="lower",
        aspect="auto",
        interpolation="nearest",
    )


def plot_fbank(fbank, title=None):
    fig, axs = plt.subplots(1, 1)
    axs.set_title(title or "Filter bank")
    axs.imshow(fbank, aspect="auto")
    axs.set_ylabel("frequency bin")
    axs.set_xlabel("mel bin")


def get_harm_bins_from_spectrogram(specgram):
    s = specgram.numpy()
    bin_avg_amp = {}
    # First, find the fundamental - this can be optimized
    # This method breaks when a harmonic is louder than the fundamental.
    for i in range(len(s[0])):
        cum = 0
        for j in s[i]:
            cum = cum + j
        bin_avg_amp[i] = cum / len(s[i])
    fund_bin = max(bin_avg_amp, key=bin_avg_amp.get)
    bins_list = [fund_bin]
    # Next guess that the bins of higher freqs are multiples of the first bin.
    for i in range(2, 8):
        bin_guess = fund_bin * i + 1
        for j in range(round(-bin_guess / 10), round(bin_guess / 10)):
            cum = 0
            bin_avg_amp = {}
            if (bin_guess + j) < 500:
                for k in s[bin_guess + j]:
                    cum = cum + k
                bin_avg_amp[bin_guess + j] = cum / len(s[bin_guess + j])
        local_max = max(bin_avg_amp, key=bin_avg_amp.get)
        bins_list.append(local_max)
    return bins_list


def get_log_amplitudes_from_bin(bins_list, specgram):
    s = specgram.numpy()
    bin_amp_dict = {}
    for i in bins_list:
        amp_list = []
        for j in s[i]:
            if j != 0:
                amp_list.append(np.log(j))
            else:
                amp_list.append(0)
        bin_amp_dict[i] = amp_list
    return bin_amp_dict


if __name__ == "__main__":
    SAMPLE_RATE = 16000
    json_dict = parse_json_metadata()
    json_dict = filter_json_metadata_pitch(json_dict)
    json_dict = filter_json_metadata_instrument_family(json_dict)
    json_dict = filter_json_metadata_quality(json_dict)

    data = NSynthDataset(json_dict, "./nsynth-train/audio", SAMPLE_RATE)

    random_number = random.randint(0, len(json_dict) - 1)
    signal, label, path, fund_freq, atk_time = data[random_number]
    print(label)

    n_fft = 1024
    win_length = None
    hop_length = 512
    n_mels = 128

    # For spectrogram
    # Define transform
    spectrogram = transforms.Spectrogram(n_fft=n_fft)
    # Perform transform
    spec = spectrogram(signal)

    mel_spectrogram = transforms.MelSpectrogram(
        sample_rate=SAMPLE_RATE,
        n_fft=n_fft,
        win_length=win_length,
        hop_length=hop_length,
        center=True,
        pad_mode="reflect",
        power=2.0,
        norm="slaney",
        n_mels=n_mels,
        mel_scale="htk",
    )

    melspec = mel_spectrogram(signal)

    mel_filters = functional.melscale_fbanks(
        int(n_fft // 2 + 1),
        n_mels=n_mels,
        f_min=0.0,
        f_max=SAMPLE_RATE / 2.0,
        sample_rate=SAMPLE_RATE,
        norm="slaney",
    )

    print("atk_time: ", atk_time)
    harm_bins = get_harm_bins_from_spectrogram(spec[0])
    print("harm_bins: ", harm_bins)
    harm_amps = get_log_amplitudes_from_bin(harm_bins, spec[0])
    print("harm amps: ", harm_amps)
    print("fundamental_freq: ", fund_freq)

    # fig, axs = plt.subplots(3, 1)
    # plot_waveform(signal, SAMPLE_RATE, title="Original waveform", ax=axs[0])
    plot_spectrogram(spec[0], title="spectrogram")
    # plot_spectrogram(melspec[0], title="MelSpectrogram - torchaudio", ylabel="mel freq", ax=axs[2])
    # plot_fbank(mel_filters, "Mel Filter Bank - torchaudio")
    # fig.tight_layout()
    plt.show()
