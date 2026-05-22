"""
Utilities for applying filters to EMG data, including handling of NaN values and loading filter configurations from JSON files.
"""

import json
import numpy as np
from pathlib import Path
from scipy.signal import butter, sosfiltfilt, filtfilt, iirnotch

CONFIG_PATH = Path(__file__).parent / "config" / "plot_config.json"

def apply_single_filter(data: np.ndarray, fs: float, filter_def: dict) -> np.ndarray:
    """Apply a single filter to the data based on the filter definition."""
    filter_type = filter_def["type"].lower()

    if filter_type == "bandpass":
        sos = butter(
            int(filter_def["order"]),
            [float(filter_def["lowcut"]), float(filter_def["highcut"])],
            btype="bandpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "highpass":
        sos = butter(
            int(filter_def["order"]),
            float(filter_def["cutoff"]),
            btype="highpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "lowpass":
        sos = butter(
            int(filter_def["order"]),
            float(filter_def["cutoff"]),
            btype="lowpass",
            fs=float(fs),
            output="sos",
        )
        return sosfiltfilt(sos, data, axis=0)

    if filter_type == "notch":
        b, a = iirnotch(
            w0=float(filter_def["freq"]),
            Q=float(filter_def["q"]),
            fs=float(fs),
        )
        return filtfilt(b, a, data, axis=0)

    raise ValueError(f"Unsupported filter type in config: {filter_type}")


def apply_filters(data: np.ndarray, fs: float, filter_list: list[dict]) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64)
    for filter_def in filter_list:
        out = apply_single_filter(out, fs, filter_def)
    return out


def load_signal_filters(config_path: Path) -> dict[str, dict]:
    """Load filter configurations from a JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config["signals"]