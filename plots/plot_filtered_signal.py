"""
Script for plotting signals from .bio files with optional filtering based on a JSON config.
"""

import json
import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path
from scipy.signal import butter, filtfilt, iirnotch, sosfiltfilt


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from utils.read_bio_file import read_bio_file


CONFIG_PATH = Path(__file__).with_name("plot_config.json")


def _load_signal_filters(config_path: Path) -> dict[str, dict]:
    """Load filter configurations from a JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return config["signals"]


# --------------------------------------------
# Filtering and NaN handling utilities
# --------------------------------------------


def _apply_single_filter(data: np.ndarray, fs: float, filter_def: dict) -> np.ndarray:
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


def _iter_true_runs(mask: np.ndarray):
    """Yield start and end indices of consecutive True runs in a boolean mask."""
    start = None
    for idx, is_true in enumerate(mask):
        if is_true and start is None:
            start = idx
        elif not is_true and start is not None:
            yield start, idx
            start = None
    if start is not None:
        yield start, mask.size


def _apply_filters_no_nan(data: np.ndarray, fs: float, filter_list: list[dict]) -> np.ndarray:
    out = np.asarray(data, dtype=np.float64)
    for filter_def in filter_list:
        out = _apply_single_filter(out, fs, filter_def)
    return out


def _apply_filters_nan(data: np.ndarray, fs: float, filter_list: list[dict]) -> np.ndarray:
    """Apply filters to data that may contain NaNs, processing only finite segments."""
    arr = np.asarray(data, dtype=np.float64)
    
    # If the array has no NaNs, apply the filters directly.
    if np.isfinite(arr).all():
        return _apply_filters_no_nan(arr, fs, filter_list)

    # Create an output array initialized with NaNs, and fill in the filtered values only for finite segments.
    out = np.full(arr.shape, np.nan, dtype=np.float64)
    for channel in range(arr.shape[1]):
        series = arr[:, channel]
        finite_mask = np.isfinite(series)
        for start, end in _iter_true_runs(finite_mask):
            segment = series[start:end].reshape(-1, 1)
            filtered = _apply_filters_no_nan(segment, fs, filter_list).reshape(-1)
            out[start:end, channel] = filtered

    return out


# --------------------------------------------
# Plot preparation utilities
# --------------------------------------------


def _compute_channel_spacing(data: np.ndarray) -> float:
    """Compute vertical spacing for plotting multiple channels."""
    q95 = np.nanpercentile(data, 95, axis=0)
    q05 = np.nanpercentile(data, 5, axis=0)
    spread = np.nanmedian(q95 - q05)
    if not np.isfinite(spread) or spread <= 0:
        return 1.0
    return float(spread * 1.5)


def _center_finite_runs(channel: np.ndarray) -> np.ndarray:
    """Center the finite values of a channel by subtracting the global mean of finite values."""
    centered = np.asarray(channel, dtype=np.float64).copy()
    finite_mask = np.isfinite(centered)
    
    if np.any(finite_mask):
        global_mean = np.mean(centered[finite_mask])
        centered[finite_mask] -= global_mean
        
    return centered


def _color_nan_regions(ax, t: np.ndarray, data: np.ndarray) -> None:
    """Highlight regions in the plot where data contains NaNs."""
    # Create a mask for rows that contain any NaN values across channels
    nan_mask = np.any(~np.isfinite(data), axis=1)
    if not np.any(nan_mask):
        return

    # Calculate the time step (dt) for x-axis spans
    dt = float(t[1] - t[0]) if t.size > 1 else 0.0

    for start, end in _iter_true_runs(nan_mask):
        x0 = float(t[start - 1]) if start > 0 else float(t[start])
        x1 = float(t[end - 1] + dt)
        ax.axvspan(x0, x1, color="red", alpha=0.18, zorder=0)


# --------------------------------------------
# Main
# --------------------------------------------

def main():

    parser = argparse.ArgumentParser(description="Plot signal from a .bio file.")
    parser.add_argument("file_path", help="Path to the .bio file")
    parser.add_argument("--filter", action="store_true", help="Apply filtering to the signals")
    args = parser.parse_args()

    file_path = args.file_path
    signal_filters = _load_signal_filters(CONFIG_PATH)

    signals = read_bio_file(file_path)

    # Apply filters if requested and if the signal is in the config
    if args.filter:
        for sig_name, sig_cfg in signal_filters.items():
            if sig_name in signals:
                signals[sig_name]["data"] = _apply_filters_nan(
                    data=signals[sig_name]["data"],
                    fs=signals[sig_name]["fs"],
                    filter_list=sig_cfg["filters"],
                )

    # Plot
    for sig_name, sig_data in signals.items():
        n_samp, n_ch = sig_data["data"].shape
        data = np.asarray(sig_data["data"], dtype=np.float64)

        t = np.arange(n_samp) / sig_data["fs"]
        fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
        fig.suptitle(sig_name)
        ax.set_xlabel("Time [s]")
        _color_nan_regions(ax, t, data)

        if n_ch > 1:
            spacing = _compute_channel_spacing(data)
            offsets = np.arange(n_ch) * spacing
            for i in range(n_ch):
                channel = _center_finite_runs(data[:, i])
                ax.plot(t, channel + offsets[i])

            ax.set_yticks(offsets)
            ax.set_yticklabels([f"Ch {i + 1}" for i in range(n_ch)])
            ax.set_ylabel("Channels")
        else:
            ax.plot(t, data[:, 0], label="Ch 1")
            ax.set_ylabel("Amplitude")
            ax.legend(loc="upper right")

    plt.show()


if __name__ == "__main__":
    main()