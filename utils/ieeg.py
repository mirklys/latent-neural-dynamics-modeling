from pathlib import Path
import mne

def band_pass_resample(
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
