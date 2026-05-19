"""
Diagnostic plot: Base signal channels vs. trigger signal with edge markers.

Usage
-----
    python plot_signal_and_trigger.py <file.bio>
    python plot_signal_and_trigger.py <file.bio> --filter
    python plot_signal_and_trigger.py <file.bio> --x_axis_hw_ts

The script looks for a base signal (defined by BASE_SIGNAL_NAME) and a trigger 
signal (defined by TRIGGER_SIGNAL_NAME) in the .bio file.
It produces a single figure with two stacked subplots:
  - Top    : all base signal channels, offset-stacked and centred.
  - Bottom : trigger signal as a step plot (raw word labels).
Green dashed lines mark rising edges (trigger on) and red dashed lines mark
falling edges (trigger off).  The word label is annotated above the upper subplot
at each rising edge.
"""

import sys
import argparse
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.read_bio_file import read_bio_file
from utils.filter import apply_filters_nan, load_signal_filters
from plot_filtered_signal import apply_channel_exclusions, plot_signal_on_axis, extract_time_axis


CONFIG_PATH = Path(__file__).parent / "config" / "plot_config.json"

BASE_SIGNAL_NAME = "emg"
TRIGGER_SIGNAL_NAME = "trigger"


def _detect_trigger_edges(
    trigger: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Detect rising and falling edges in a 1-D trigger signal.

    The trigger value encodes the word label (non-zero = active window, 0 = idle).

    Returns
    -------
    rising_idx    : sample indices of rising edges
    rising_words  : word label at each rising edge
    falling_idx   : sample indices of falling edges
    falling_words : word label immediately before each falling edge
    """
    trig = np.asarray(trigger, dtype=np.float64).ravel()
    active = (trig != 0).astype(np.int8)
    diff = np.diff(active, prepend=active[0])

    rising_idx = np.where(diff > 0)[0]
    falling_idx = np.where(diff < 0)[0]

    rising_words = trig[rising_idx].astype(int)
    falling_words = trig[np.maximum(falling_idx - 1, 0)].astype(int)

    return rising_idx, rising_words, falling_idx, falling_words


def plot_base_signal_and_trigger_overview(
    base_sig: dict,
    trigger_sig: dict,
    filename: str,
    mic_sig: dict | None = None,
    ts_base: dict | None = None,
    ts_mic: dict | None = None,
    use_hw_ts: bool = False,
) -> Figure:
    """Build and return the base signal and trigger figure."""

    trig_raw = np.asarray(trigger_sig["data"], dtype=np.float64)
    if trig_raw.ndim == 2:
        trig_raw = trig_raw[:, 0]
    trig_fs = float(trigger_sig["fs"])

    # Build time axes
    t_base, len_base = extract_time_axis(base_sig, ts_base, use_hw_ts)
    t_trig = np.arange(trig_raw.shape[0]) / trig_fs  # trigger has no hw timestamp

    # When using hw timestamps, shift all axes so they start from 0
    t_offset = t_base[0] if use_hw_ts and len(t_base) > 0 else 0.0
    t_base = t_base - t_offset

    # Build mic time axis
    t_mic: np.ndarray = np.empty(0)
    len_mic = 0
    if mic_sig is not None:
        t_mic, len_mic = extract_time_axis(mic_sig, ts_mic, use_hw_ts)
        t_mic = t_mic - t_offset

    # Detect edges on trigger signal
    rising_idx, rising_words, falling_idx, _ = _detect_trigger_edges(trig_raw)
    rising_t = t_trig[rising_idx]
    falling_t = t_trig[falling_idx]

    # For base signal subplot, only annotate edges within base signal time range
    base_edge_mask = (rising_t >= t_base[0]) & (rising_t <= t_base[-1])
    rising_t_base = rising_t[base_edge_mask]
    rising_words_base = rising_words[base_edge_mask]
    falling_t_base = falling_t[(falling_t >= t_base[0]) & (falling_t <= t_base[-1])]

    # Edge masks for mic subplot
    rising_t_mic: np.ndarray = np.empty(0)
    falling_t_mic: np.ndarray = np.empty(0)
    if mic_sig is not None and len(t_mic) > 0:
        mic_edge_mask = (rising_t >= t_mic[0]) & (rising_t <= t_mic[-1])
        rising_t_mic = rising_t[mic_edge_mask]
        falling_t_mic = falling_t[(falling_t >= t_mic[0]) & (falling_t <= t_mic[-1])]

    ax_mic = None
    if mic_sig is None:
        fig, (ax_base, ax_trig) = plt.subplots(
            2,
            1,
            figsize=(18, 8),
            layout="constrained",
            sharex=True,
            gridspec_kw={"height_ratios": [2, 2]},
        )
    else:
        fig, (ax_base, ax_mic, ax_trig) = plt.subplots(
            3, 1, figsize=(18, 10), layout="constrained", sharex=True,
            gridspec_kw={"height_ratios": [2, 2, 2]},
        )

    fig.suptitle(filename, fontsize=14)

    plot_signal_on_axis(ax_base, BASE_SIGNAL_NAME, base_sig, t_base, len_base)

    if mic_sig is not None and ax_mic is not None:
        plot_signal_on_axis(ax_mic, "mic_emg", mic_sig, t_mic, len_mic)

    for rt, word in zip(rising_t_base, rising_words_base):
        ax_base.axvline(rt, color="green", lw=0.9, ls="--", alpha=0.8, zorder=3)
        ax_base.text(
            rt, 1.01, f"W{word}",
            transform=ax_base.get_xaxis_transform(),
            fontsize=7, color="green", ha="center", va="bottom", clip_on=False,
        )
    for ft in falling_t_base:
        ax_base.axvline(ft, color="red", lw=0.9, ls="--", alpha=0.8, zorder=3)

    if mic_sig is not None and ax_mic is not None:
        for rt in rising_t_mic:
            ax_mic.axvline(rt, color="green", lw=0.9, ls="--", alpha=0.8, zorder=3)
        for ft in falling_t_mic:
            ax_mic.axvline(ft, color="red", lw=0.9, ls="--", alpha=0.8, zorder=3)

    # Trigger
    ax_trig.step(t_trig, trig_raw, where="post", color="steelblue", lw=1.2)
    ax_trig.set_ylabel("Word label")
    ax_trig.set_xlabel("Time [s]")
    ax_trig.set_title(TRIGGER_SIGNAL_NAME)

    for rt in rising_t:
        ax_trig.axvline(rt, color="green", lw=0.9, ls="--", alpha=0.8, zorder=3)
    for ft in falling_t:
        ax_trig.axvline(ft, color="red", lw=0.9, ls="--", alpha=0.8, zorder=3)

    fig.legend(
        handles=[
            Line2D([0], [0], color="green", ls="--", lw=1.2, label="Trigger on"),
            Line2D([0], [0], color="red",   ls="--", lw=1.2, label="Trigger off"),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.8,
    )

    return fig


# --------------------------------------------
# Main
# --------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description=f"Plot {BASE_SIGNAL_NAME.upper()} vs. trigger edge alignment from a .bio file."
    )
    parser.add_argument("file_path", help="Path to the .bio file")
    parser.add_argument("--filter", action="store_true", help=f"Apply filtering to the {BASE_SIGNAL_NAME.upper()} signal")
    parser.add_argument("x_axis_hw_ts", action="store_true", help="Use hardware timestamps for the x-axis when available")
    args = parser.parse_args()

    signal_filters = load_signal_filters(CONFIG_PATH)
    signals = read_bio_file(args.file_path)

    filename = Path(args.file_path).name

    base_key = next((k for k in signals if k.lower() == BASE_SIGNAL_NAME.lower()), None)
    trigger_key = next((k for k in signals if k.lower() == TRIGGER_SIGNAL_NAME.lower()), None)
    mic_key = next((k for k in signals if k.lower() == "mic_emg"), None)

    if not base_key:
        print(f"[Error] No {BASE_SIGNAL_NAME.upper()} signal found in file. Exiting.")
        sys.exit(1)

    if not trigger_key:
        print(f"[Error] No trigger signal found (expected key: '{TRIGGER_SIGNAL_NAME}'). Exiting.")
        sys.exit(1)

    if args.filter:
        base_cfg = signal_filters.get(base_key, {})
        filter_list = base_cfg.get("filters")
        if filter_list:
            signals[base_key]["data"] = apply_filters_nan(
                data=signals[base_key]["data"],
                fs=signals[base_key]["fs"],
                filter_list=filter_list,
            )
        else:
            print(f"[Warning] --filter requested but no filter config found for '{base_key}'.")
        # Note: do not apply filtering to mic_emg (kept raw)

    apply_channel_exclusions(signals, signal_filters)

    mic_sig = signals[mic_key] if mic_key else None
    plot_base_signal_and_trigger_overview(
        signals[base_key],
        signals[trigger_key],
        filename,
        mic_sig=mic_sig,
        ts_base=signals.get(f"timestamp_{base_key}"),
        ts_mic=signals.get(f"timestamp_{mic_key}") if mic_key else None,
        use_hw_ts=args.x_axis_hw_ts,
    )
    plt.show()


if __name__ == "__main__":
    main()