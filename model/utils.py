import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal
from scipy.io import wavfile

EPS = np.finfo(float).eps
C = 343  # m / s


def read(path: str | Path) -> tuple[int, np.ndarray]:
    sr, x = wavfile.read(path)
    x = x / 2**15

    return sr, x


def _pad(x: np.ndarray, dft_size: int = 512, hop_size: int = 128) -> np.ndarray:
    return (
        np.pad(x, (int(np.ceil((hop_size - d) / 2)), int(np.floor((hop_size - d) / 2))))
        if (d := (x.size - dft_size) % hop_size)
        else x
    )


def _calculate_figure_size(
    nrows: int = 1,
    ncols: int = 1,
    subplot_width: int = 3,
    subplot_height: int = 2,
    min_spacing: float = 0.5,
) -> tuple[float, float]:
    return (ncols * subplot_width) + (ncols - 1) * min_spacing, (
        nrows * subplot_height
    ) + (nrows - 1) * min_spacing


def stft(
    x: np.ndarray,
    dft_size: int = 512,
    hop_size: int = 128,
    window: np.ndarray | bool = True,
) -> np.ndarray:
    x = _pad(x, dft_size, hop_size)

    if isinstance(window, bool):
        if window == True:
            window = signal.windows.hamming(dft_size, sym=False)
        elif window == False:
            window = np.ones(dft_size)

    # Return a complex-valued spectrogram (frequencies x time)
    return np.array(
        [
            np.fft.rfft(window * x[i * hop_size : i * hop_size + dft_size])
            for i in np.arange((x.size - dft_size) // hop_size)  # Ew...
        ]
    ).T


def istft(
    S: np.ndarray,
    dft_size: int = 512,
    hop_size: int = 128,
    window: np.ndarray | bool = True,
) -> np.ndarray:
    _, n_frames = S.shape

    if isinstance(window, bool):
        if window == True:
            window = signal.windows.hamming(dft_size, sym=False)
        elif window == False:
            window = np.ones(dft_size)

    x = np.zeros(n_frames * hop_size + dft_size)
    for i in np.arange(n_frames):
        x[i * hop_size : i * hop_size + dft_size] += np.fft.irfft(S[:, i]) * window

    return x


def sound(x: np.ndarray, sr: int, label="") -> None:
    from IPython.display import display, Audio, HTML

    display(
        HTML(
            "<style> table, th, td {border: 0px; }</style> <table><tr><td>"
            + label
            + "</td><td>"
            + Audio(x, rate=sr)._repr_html_()[3:]
            + "</td></tr></table>"
        )
    )


def specshow(
    data: np.ndarray,
    sr: int,
    dft_size: int = 512,
    hop_size: int = 128,
    ax: Axes | None = None,
    title: str = "",
    label_axis=True,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    ax.pcolormesh(
        np.arange(0, data.shape[1] * hop_size, hop_size) / sr,
        np.fft.rfftfreq(dft_size, 1.0 / sr),
        np.log(np.abs(data) + EPS),
        cmap="magma",
    )

    ax.set_title(title)

    if label_axis:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Frequency (Hz)")

    return ax


def waveshow(
    x: np.ndarray,
    ax: Axes | None = None,
    title: str = "",
    label_axis: bool = True,
    limit_axes: tuple[tuple[float, float], tuple[float, float]] | None = None,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    ax.plot(x)

    ax.set_title(title)

    if label_axis:
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

    if limit_axes:
        ax.set_xlim(limit_axes[0])
        ax.set_ylim(limit_axes[1])

    return ax


def waveshows(
    xs: list[np.ndarray],
    titles: list[str] = [],
    suptitle: str = "",
    limit_axis: list | tuple | None = None,
) -> tuple[Figure, np.ndarray]:
    n = len(xs)
    fig, axes = plt.subplots(n, figsize=_calculate_figure_size(n))

    if isinstance(axes, Axes):
        axes = np.array([axes])

    if limit_axis is None:
        limit_axis = [None] * len(xs)
    elif isinstance(limit_axis, tuple):
        limit_axis = [limit_axis] * len(xs)

    titles.extend([""] * (min(n, axes.size) - len(titles)))

    for ax, x, title, limit in zip(axes, xs, titles, limit_axis):
        waveshow(x.T if x.ndim == 2 else x, ax, title, False, limit)

    fig.suptitle(suptitle)
    fig.supxlabel("Samples")
    fig.supylabel("Amplitude")
    fig.tight_layout()

    return fig, axes


def freqshow(
    w: np.ndarray,
    h: np.ndarray,
    sr: int,
    cf: int | list[int] | None = None,
    ax: Axes | None = None,
    title: str = "",
    hz_db: bool = True,
    label_axis: bool = True,
    limit_axis: bool = True,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    if hz_db:
        ax.plot(w * sr / (2 * np.pi), 20 * np.log10(np.abs(h)))
    else:
        ax.plot(w, np.abs(h))

    if cf is None:
        cf = []
    elif isinstance(cf, int):
        cf = [cf]

    for c in cf:
        ax.axvline(c, color="red")

    ax.grid(which="both", axis="both")
    ax.set_title(title)

    if label_axis:
        ax.set_ylabel(f"Amplitude{'(dB)' if hz_db else ''}")
        ax.set_xlabel(f"Frequency{'(Hz)' if hz_db else ''}")

    if limit_axis:
        xlim, ylim = ((0, sr / 2), (-100, 50)) if hz_db else ((0, np.pi), (-2, 2))
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    return ax


def freqshows(
    wh: list[tuple[np.ndarray, np.ndarray]],
    sr: int | list[int],
    cf: int | list[int] | list[list[int]] | None = None,
    titles: list[str] = [],
    suptitle: str = "",
) -> tuple[Figure, np.ndarray]:
    n = len(wh)
    if isinstance(sr, int):
        sr = [sr] * n

    sharex_arg = "all" if len(sr) == 1 else "none"

    fig, axes = plt.subplots(n, sharex=sharex_arg, figsize=_calculate_figure_size(n, 1))

    if isinstance(axes, Axes):
        axes = np.array([axes])

    if cf is None:
        cf = [[] for _ in range(n)]
    elif isinstance(cf, int):
        cf = [cf] * n

    titles.extend([""] * (min(n, axes.size) - len(titles)))

    for ax, (w, h), s, c, title in zip(axes, wh, sr, cf, titles):
        freqshow(
            w,
            h,
            s,
            c,
            ax,
            title,
            label_axis=False,
        )

    fig.suptitle(suptitle)
    fig.supxlabel("Frequency (Hz)")
    fig.supylabel("Amplitude (dB)")
    fig.tight_layout()

    return fig, axes


def specshows(
    Ss: list[np.ndarray],
    sr: int | list[int],
    dft_size: int | list[int] = 512,
    hop_size: int | list[int] = 128,
    titles: list[str] = [],
    suptitle: str = "",
) -> tuple[Figure, np.ndarray]:
    n = len(Ss)
    share_axis_arg = "all" if len({S.shape for S in Ss}) == 1 else "none"

    if isinstance(sr, int):
        sr = [sr] * n
    if isinstance(dft_size, int):
        dft_size = [dft_size] * n
    if isinstance(hop_size, int):
        hop_size = [hop_size] * n

    assert (
        len(sr) == len(dft_size) == len(hop_size) == n
    ), f"Expected {n} of each sample rate, DFT size, and hop size, instead got {len(sr), len(dft_size), len(hop_size)}"

    nrows, ncols = max(1, np.ceil(n / 2).astype(int)), 1 if n <= 1 else 2
    fig, axes = plt.subplots(
        nrows,
        ncols,
        sharex=share_axis_arg,
        sharey=share_axis_arg,
        figsize=_calculate_figure_size(nrows, ncols),
    )

    titles.extend([""] * (n - len(titles)))

    for S, sr_, dft_size_, hop_size_, ax, t in zip(
        Ss, sr, dft_size, hop_size, axes.flatten(), titles
    ):
        specshow(S, sr_, dft_size_, hop_size_, ax, title=t, label_axis=False)

    fig.suptitle(suptitle)
    fig.supxlabel("Time (s)")
    fig.supylabel("Frequency (Hz)")
    fig.tight_layout()

    return fig, axes


def stemshow(
    x: np.ndarray,
    ax: Axes | None = None,
    title: str = "",
    label_axis=True,
) -> Axes:
    if ax is None:
        ax = plt.gca()

    ax.stem(x, markerfmt=".")
    ax.set_title(title)

    if label_axis:
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")

    return ax


def impulse(n: int) -> np.ndarray:
    x = np.zeros(n)
    x[0] = 1
    return x


def impulseshow(
    b: np.ndarray,
    a: np.ndarray | int,
    sr: int,
    ax: Axes | None = None,
    title: str = "",
    label_axis=True,
    playback=False,
) -> Axes:
    x = signal.lfilter(b, a, impulse(sr))[
        : 4 * max(b.size, a.size if isinstance(a, np.ndarray) else 1)
    ]
    assert isinstance(x, np.ndarray)
    stem = stemshow(x, ax, title, label_axis)

    if ax is None:
        plt.show()
    if playback:
        sound(x, sr, title)

    return stem


def impulseshows(
    ba: list[tuple[np.ndarray, np.ndarray | int]],
    sr: int | list[int],
    titles: list[str] = [],
    suptitle: str = "",
    label_axis=True,
    playback=False,
) -> tuple[Figure, np.ndarray]:
    n = len(ba)
    sharex_arg = "none"

    if isinstance(sr, int):
        sharex_arg = "all"
        sr = [sr] * n

    fig, axes = plt.subplots(n, sharex=sharex_arg, figsize=_calculate_figure_size(n))

    if isinstance(axes, Axes):
        axes = np.array([axes])

    titles.extend([""] * (min(n, axes.size) - len(titles)))

    for ax, (b, a), s, title in zip(axes, ba, sr, titles):
        impulseshow(b, a, s, ax, title, False, playback)

    if label_axis:
        fig.supxlabel("Samples")
        fig.supylabel("Amplitude")

    fig.suptitle(suptitle)
    fig.tight_layout()

    return fig, axes


def sounds(
    sounds: list[np.ndarray], sr: int | list[int], labels: list[str] = [], title=""
) -> None:
    if title:
        print(title)

    labels.extend([""] * (len(sounds) - len(labels)))

    if isinstance(sr, int):
        sr = [sr] * len(sounds)

    for x, s, l in zip(sounds, sr, labels):
        sound(x, sr=s, label=l)


def metres_to_samples(d: float | np.ndarray, sr: int) -> int | np.ndarray:
    return np.round(d / C * sr).astype(int)
