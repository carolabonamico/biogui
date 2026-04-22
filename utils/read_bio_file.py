"""
File reading utilities for .bio files.
"""


from __future__ import annotations
import struct
import numpy as np
import sys


def _print_signal_info(signals: dict) -> None:
    """Print basic information about the signals in the .bio file."""
    print("Signals information:")
    for sig_name, sig_data in signals.items():
        data = sig_data["data"]
        print(
            f"{sig_name}: fs={sig_data['fs']}, n_samp={data.shape[0]}, "
            f"n_ch={data.shape[1]}, dtype={data.dtype}"
        )


def read_bio_file(file_path: str) -> dict:

    """
    Read a .bio file and extract all signals, timestamps, and triggers.
    """
    dtypeMap = {
        "?": np.dtype("bool"),
        "b": np.dtype("int8"),
        "B": np.dtype("uint8"),
        "h": np.dtype("int16"),
        "H": np.dtype("uint16"),
        "i": np.dtype("int32"),
        "I": np.dtype("uint32"),
        "q": np.dtype("int64"),
        "Q": np.dtype("uint64"),
        "f": np.dtype("float32"),
        "d": np.dtype("float64"),
    }

    with open(file_path, "rb") as f:
        n_signals = struct.unpack("<I", f.read(4))[0]
        fs_base, n_samp_base = struct.unpack("<fI", f.read(8))

        signals = {}
        for _ in range(n_signals):
            sig_name_len = struct.unpack("<I", f.read(4))[0]
            sig_name = struct.unpack(f"<{sig_name_len}s", f.read(sig_name_len))[
                0
            ].decode()
            fs, n_samp, n_ch, dtype = struct.unpack("<f2Ic", f.read(13))

            signals[sig_name] = {
                "fs": fs,
                "n_samp": n_samp,
                "n_ch": n_ch,
                "dtype": dtypeMap[dtype.decode("ascii")],
            }

        is_trigger = struct.unpack("<?", f.read(1))[0]
        # is_trigger_str = struct.unpack("<?", f.read(1))[0]
        # print(f"Trigger set to: {is_trigger}")
        # print(f"Trigger str set to: {is_trigger_str}")

        # 1. Timestamp
        ts = np.frombuffer(f.read(8 * n_samp_base), dtype=np.float64).reshape(
            n_samp_base, 1
        )
        signals["timestamp"] = {"data": ts, "fs": fs_base}

        # 2. Signals data
        for sig_name, sig_data in signals.items():
            if sig_name == "timestamp":
                continue

            n_samp = sig_data.pop("n_samp")
            n_ch = sig_data.pop("n_ch")
            dtype = sig_data.pop("dtype")

            data = np.frombuffer(
                f.read(dtype.itemsize * n_samp * n_ch), dtype=dtype
            ).reshape(n_samp, n_ch)
            sig_data["data"] = data

        # 3. Trigger
        if is_trigger:
            itemsize = 4    # saving as uint32_t
            trigger = np.frombuffer(f.read(itemsize * n_samp_base), dtype=np.uint32).reshape(n_samp_base, 1)
            signals["trigger"] = {"data": trigger, "fs": fs_base}

        # # 4. Trigger string (len-prefixed UTF-8 per sample)
        # if is_trigger_str:
        #     trigger_str = []
        #     for _ in range(n_samp_base):
        #         (L,) = struct.unpack("<I", f.read(4))
        #         if L == 0:
        #             trigger_str.append("")
        #         else:
        #             b = f.read(L)
        #             trigger_str.append(b.decode("utf-8", errors="replace"))

        #     # store as (n_samp_base, 1) like other signals
        #     signals["trigger_str"] = {
        #         "data": np.array(trigger_str, dtype=object).reshape(n_samp_base, 1),
        #         "fs": fs_base,
        #     }
    
    # _print_signal_info(signals)

    return signals

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_bio_file>")
        sys.exit(1)

    signals = read_bio_file(sys.argv[1])