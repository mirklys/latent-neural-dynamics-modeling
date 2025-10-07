from pathlib import Path
import mne

from scipy.signal import butter, filtfilt
import numpy as np


def preprocess_ieeg(
    ieeg_headers_file: str, sfreq: int, low_freq: int, high_freq: int, notch_freqs
) -> dict[str, list[float]] | None:
    ieeg_path = Path(ieeg_headers_file)

    if not ieeg_path.exists():
        return None

    try:
        raw = mne.io.read_raw_brainvision(ieeg_path, preload=True, verbose=False)

        raw.notch_filter(freqs=notch_freqs, verbose=False)
        raw.filter(l_freq=low_freq, h_freq=high_freq)
        raw.resample(sfreq=sfreq, verbose=False)

        data = raw.get_data()
        channels_data = {ch: d.tolist() for ch, d in zip(raw.ch_names, data)}
        channels_data["sfreq"] = float(sfreq)
        return channels_data
    except Exception as e:
        from utils.logger import get_logger

        logger = get_logger()

        logger.info(str(e))
        return None


def filter_recording(
    recording: list[float],
    low_freq: int,
    high_freq: int,
    notch_freqs: list[int],
    sfreq: int,
) -> list[float]:
    recording_ = np.array(recording, dtype=np.float64)

    recording_ = mne.filter.filter_data(
        data=recording_, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq, verbose=False
    )
    recording_ = mne.filter.notch_filter(
        x=recording_, Fs=sfreq, freqs=notch_freqs, verbose=False
    )

    return list(recording_)


def epoch_trials(
    recording: list[float],
    window_size: int = 500,
    step_size: int = 250,
) -> list[list[float]]:
    n_samples = len(recording)
    recording_ = np.array(recording, dtype=np.float64)

    epochs = []
    for start in range(0, n_samples, step_size):
        end = start + window_size
        if end > n_samples:
            last_epoch = recording_[start:]
            padding = np.zeros(end - n_samples)
            epoch = np.concatenate([last_epoch, padding])
        else:
            epoch = recording_[start:end]
        epochs.append(epoch.tolist())

    return epochs
