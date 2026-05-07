"""Align and repair signals stored in a .bio file."""

from __future__ import annotations

import sys
import numpy as np
from pathlib import Path
from read_bio_file import read_bio_file
from write_bio_file import write_bio_file
from check_packet_loss import compute_modulus, unwrap_signal


def _trim_signals(signals: dict) -> dict:
    """Trim all signals to the common time window defined by hardware timestamps."""
    hw_timestamps_names = [
        name for name in signals
        if name.startswith("timestamp_") and signals[name]["data"].size > 0
    ]

    if not hw_timestamps_names:
        return signals

    # Unwraps the hardware timestamps
    unwrapped_timestamps = {}
    for ts_name in hw_timestamps_names:
        ts_raw = signals[ts_name]["data"].reshape(-1)
        if ts_raw.size == 0:
            continue
        ts_modulus = compute_modulus(ts_raw, ts_raw.dtype)
        unwrapped_timestamps[ts_name] = unwrap_signal(ts_raw, ts_modulus).astype(np.float64)

    if not unwrapped_timestamps:
        return signals

    # Computes the common time window across unwrapped hardware timestamps
    starts = [float(unwrapped_timestamps[t][0]) for t in unwrapped_timestamps]
    ends = [float(unwrapped_timestamps[t][-1]) for t in unwrapped_timestamps]
    common_start = max(starts)
    common_end = min(ends)

    if common_start >= common_end:
        raise ValueError("Signals have no overlapping time window.")

    # Computes the trimming indices for hardware timestamps and their associated signals
    packet_windows = {}
    for ts_name in hw_timestamps_names:
        ts_data = unwrapped_timestamps.get(ts_name)
        if ts_data is None or ts_data.size == 0:
            continue
        start_packet = int(np.searchsorted(ts_data, common_start, side="left"))
        end_packet = int(np.searchsorted(ts_data, common_end, side="right"))
        packet_windows[ts_name] = (start_packet, end_packet)

    # Applies the trimming to hardware signals
    trimmed = {}
    for signal_name, signal in signals.items():
        # Ignoring the software signals
        if signal_name in ("timestamp", "trigger"):
            trimmed[signal_name] = {"fs": signal["fs"], "data": signal["data"].copy()}
            continue

        # Gets the name of the signal associated with the hw timestamp
        if signal_name.startswith("timestamp_"):
            ts_name = signal_name
        elif signal_name.startswith("counter_"):
            ts_name = f"timestamp_{signal_name.replace('counter_', '', 1)}"
        else:
            ts_name = f"timestamp_{signal_name}"

        # If the associated timestamp is not among the valid HW ones, it is copied without trimming
        if ts_name not in packet_windows:
            trimmed[signal_name] = {"fs": signal["fs"], "data": signal["data"].copy()}
            continue

        start_packet, end_packet = packet_windows[ts_name]
        samples_per_packet = int(round(signal["fs"] / signals[ts_name]["fs"]))

        sample_start = start_packet * samples_per_packet
        sample_end = end_packet * samples_per_packet
        trimmed_data = signal["data"][sample_start:sample_end].copy()

        trimmed[signal_name] = {"fs": signal["fs"], "data": trimmed_data}

    return trimmed


def _repair_counter_losses(signals: dict) -> None:
    """Repair counter signals by filling in missing packets with NaNs, reconstructs the correct counter values and timestamps."""
    counter_names = [name for name in signals if name.startswith("counter_")]

    for counter_name in counter_names:
        # Derives the associated payload and timestamp signal names
        signal_name = counter_name.replace("counter_", "", 1)
        timestamp_name = f"timestamp_{signal_name}"

        if signal_name not in signals or timestamp_name not in signals:
            continue

        counter_raw = signals[counter_name]["data"].reshape(-1)
        if counter_raw.size == 0:
            continue

        counter_original_dtype = counter_raw.dtype

        modulus = compute_modulus(counter_raw, counter_original_dtype)
        unwrapped = unwrap_signal(counter_raw, modulus)

        relative_indices = (unwrapped - unwrapped[0]).astype(np.int64)
        total_expected_packets = int(relative_indices[-1]) + 1

        payload = signals[signal_name]["data"].astype(np.float64)
        payload_channels = payload.shape[1]
        samples_per_packet = int(round(signals[signal_name]["fs"] / signals[counter_name]["fs"]))
        payload_packets = payload.reshape(len(unwrapped), samples_per_packet, payload_channels)

        rebuilt_counter = np.arange(total_expected_packets) % modulus

        rebuilt_payload_packets = np.full((total_expected_packets, samples_per_packet, payload_channels), np.nan)
        rebuilt_payload_packets[relative_indices] = payload_packets

        # Reconstruct the hardware timestamps
        timestamp_raw = signals[timestamp_name]["data"].reshape(-1)
        timestamp_original_dtype = timestamp_raw.dtype
        timestamp_modulus = compute_modulus(timestamp_raw, timestamp_original_dtype)
        timestamp_unwrapped = unwrap_signal(timestamp_raw, timestamp_modulus).astype(np.float64)

        timestamp_step = 1_000_000.0 / float(signals[timestamp_name]["fs"])
        rebuilt_timestamps = timestamp_unwrapped[0] + (np.arange(total_expected_packets, dtype=np.float64) * timestamp_step)
        rebuilt_timestamps[relative_indices] = timestamp_unwrapped
        rebuilt_timestamps_wrapped = (rebuilt_timestamps % timestamp_modulus).astype(timestamp_original_dtype)

        signals[counter_name]["data"] = rebuilt_counter.astype(counter_original_dtype).reshape(-1, 1)
        signals[signal_name]["data"] = rebuilt_payload_packets.reshape(-1, payload_channels)
        signals[timestamp_name]["data"] = rebuilt_timestamps_wrapped.reshape(-1, 1)


def align_bio_signals(file_path: str) -> dict:
    signals = read_bio_file(file_path)
    aligned_signals = _trim_signals(signals)
    _repair_counter_losses(aligned_signals)
    return aligned_signals


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("Usage: python utils/align_bio_signals.py PATH_TO_INPUT_BIO PATH_TO_OUTPUT_DIR")

    input_bio  = Path(sys.argv[1])
    output_dir = Path(sys.argv[2])

    output_dir.mkdir(parents=True, exist_ok=True)
    output_bio = output_dir / f"{input_bio.stem}_aligned{input_bio.suffix}"

    aligned = align_bio_signals(str(input_bio))
    write_bio_file(str(output_bio), aligned)
    print(f"[SAVED]: {output_bio}")