"""
Script for plotting signals from .bio files with optional filtering based on a JSON config.
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
    
from utils.read_bio_file import read_bio_file
from utils.filter import apply_filters_nan, load_signal_filters, iter_true_runs

CONFIG_PATH = Path(__file__).parent/"config"/"plot_config.json"

LINE_WIDTH = 0.8


# --------------------------------------------
# Utilities
# --------------------------------------------
        

def _get_kept_channel_ids(n_ch: int, exclude_channels: list[int]) -> list[int]:
    """Return channel indices that should be kept after applying exclusions."""
    excluded = set()
    for channel_idx in exclude_channels:
        try:
            idx = int(channel_idx)
        except (TypeError, ValueError):
            continue
        if 0 <= idx < n_ch:
            excluded.add(idx)

    return [idx for idx in range(n_ch) if idx not in excluded]


def apply_channel_exclusions(signals: dict[str, dict], signal_cfg: dict[str, dict]) -> None:
    """Apply channel exclusions from config to every matching signal."""
    for sig_name, sig_data in signals.items():
        data = np.asarray(sig_data["data"])
        if data.ndim != 2:
            continue

        cfg = signal_cfg.get(sig_name, {})
        exclude_channels = cfg.get("exclude_channels")
        if not exclude_channels:
            sig_data["channel_ids"] = list(range(data.shape[1]))
            continue

        kept_ids = _get_kept_channel_ids(data.shape[1], exclude_channels)
        sig_data["data"] = data[:, kept_ids]
        sig_data["channel_ids"] = kept_ids


# --------------------------------------------
# Plot preparation utilities
# --------------------------------------------


def compute_channel_spacing(data: np.ndarray) -> float:
    """Compute vertical spacing for plotting multiple channels."""
    q95 = np.nanpercentile(data, 95, axis=0)
    q05 = np.nanpercentile(data, 5, axis=0)
    spread = np.nanmedian(q95 - q05)
    if not np.isfinite(spread) or spread <= 0:
        return 1.0
    return float(spread * 1.5)


def center_finite_runs(channel: np.ndarray) -> np.ndarray:
    """Center the finite values of a channel by subtracting the global mean of finite values."""
    centered = np.asarray(channel, dtype=np.float64).copy()
    finite_mask = np.isfinite(centered)
    if np.any(finite_mask):
        global_mean = np.mean(centered[finite_mask])
        centered[finite_mask] -= global_mean
    return centered


def color_nan_regions(ax, t: np.ndarray, data: np.ndarray) -> None:
    """Highlight regions in the plot where data contains NaNs."""
    nan_mask = np.any(~np.isfinite(data), axis=1)
    if not np.any(nan_mask):
        return

    dt = float(t[1] - t[0]) if t.size > 1 else 0.0
    for start, end in iter_true_runs(nan_mask):
        x0 = float(t[start - 1]) if start > 0 else float(t[start])
        x1 = float(t[end - 1] + dt)
        ax.axvspan(x0, x1, color="red", alpha=0.18, zorder=0)


def extract_time_axis(
    sig_data: dict,
    ts_entry: dict | None,
    use_hw_ts: bool = False,
) -> tuple[np.ndarray, int]:
    """Build the time axis in seconds for a signal.

    If *use_hw_ts* is True and a valid ts_entry is present, the hardware
    timestamps (in µs) are repeated to match the signal sample rate.
    Otherwise a synthetic axis is built from the sample index and fs.
    """
    n_samp = sig_data["data"].shape[0]

    if use_hw_ts and ts_entry and ts_entry.get("data") is not None and ts_entry.get("fs"):
        ts_arr = np.asarray(ts_entry["data"]).ravel()
        spp = max(1, round(sig_data["fs"] / ts_entry["fs"]))
        t = np.repeat(ts_arr, spp) / 1_000_000.0  # µs → s
    else:
        t = np.arange(n_samp) / sig_data["fs"]

    min_len = min(n_samp, len(t))
    return t[:min_len], min_len


def plot_signal_on_axis(
    ax,
    sig_name: str,
    sig_data: dict,
    t: np.ndarray,
    min_len: int,
    line_width: float = LINE_WIDTH,
) -> None:
    """Plot the signal on a provided Matplotlib axis."""
    data = np.asarray(sig_data["data"], dtype=np.float64)[:min_len]
    n_ch = data.shape[1]
    channel_ids = sig_data.get("channel_ids", list(range(n_ch)))
    if len(channel_ids) != n_ch:
        channel_ids = list(range(n_ch))

    ax.set_title(sig_name)
    color_nan_regions(ax, t, data)

    if n_ch > 1:
        spacing = compute_channel_spacing(data)
        offsets = np.arange(n_ch) * spacing
        for i in range(n_ch):
            channel = center_finite_runs(data[:, i])
            ax.plot(t, channel + offsets[i], lw=line_width)
        ax.set_yticks(offsets)
        ax.set_yticklabels([f"Ch {channel_ids[i]}" for i in range(n_ch)])
        ax.set_ylabel("Channels")
    else:
        color = "C1" if sig_name.startswith("mic_") else "C0"
        ax.plot(t, data[:, 0], label=f"Ch {channel_ids[0]}", lw=line_width, color=color)
        ax.set_ylabel("Amplitude")
        ax.legend(loc="upper right")

    ax.grid(True, linestyle="-", color="#e0e0e0", linewidth=0.5, alpha=0.7)


# --------------------------------------------
# Main
# --------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Plot signal from a .bio file.")
    parser.add_argument("file_path", help="Path to the .bio file")
    parser.add_argument("--filter", action="store_true", help="Apply filtering to the signals")
    parser.add_argument(
        "--x_axis_hw_ts",
        action="store_true",
        help="Use hardware timestamps for the x-axis when available (applies to all signals)",
    )
    args = parser.parse_args()

    file_path = args.file_path
    filename = Path(file_path).name
    signal_filters = load_signal_filters(CONFIG_PATH)
    signals = read_bio_file(file_path)

    if args.filter:
        for sig_name, sig_cfg in signal_filters.items():
            filter_list = sig_cfg.get("filters")
            if sig_name in signals and filter_list:
                signals[sig_name]["data"] = apply_filters_nan(
                    data=signals[sig_name]["data"],
                    fs=signals[sig_name]["fs"],
                    filter_list=filter_list,
                )

    apply_channel_exclusions(signals, signal_filters)

    use_hw_ts = args.x_axis_hw_ts

    # Plot each signal individually
    for sig_name, sig_data in signals.items():
        if sig_name.startswith("timestamp_"):
            continue
        fig, ax = plt.subplots(figsize=(16, 6), layout="constrained")
        fig.suptitle(filename, fontsize=12)
        ts_entry = signals.get(f"timestamp_{sig_name}")
        t, min_len = extract_time_axis(sig_data, ts_entry, use_hw_ts)
        plot_signal_on_axis(ax, sig_name, sig_data, t, min_len)
        ax.set_xlabel("Time [s]")

    # Additional plot: top = base signal, bottom = matching mic_<base> signal
    for mic_name, mic_data in signals.items():
        if not mic_name.startswith("mic_"):
            continue

        base_name = mic_name[4:]
        if base_name not in signals:
            continue

        base_data = signals[base_name]
        fig, (ax_top, ax_bottom) = plt.subplots(
            2, 1, figsize=(16, 8), layout="constrained", sharex=True,
        )
        fig.suptitle(filename, fontsize=12)

        ts_base = signals.get(f"timestamp_{base_name}")
        t_base, len_base = extract_time_axis(base_data, ts_base, use_hw_ts)
        plot_signal_on_axis(ax_top, base_name, base_data, t_base, len_base)

        ts_mic = signals.get(f"timestamp_{mic_name}")
        t_mic, len_mic = extract_time_axis(mic_data, ts_mic, use_hw_ts)
        plot_signal_on_axis(ax_bottom, mic_name, mic_data, t_mic, len_mic)

        ax_bottom.set_xlabel("Time [s]")

    plt.show()


if __name__ == "__main__":
    main()