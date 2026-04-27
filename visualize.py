import argparse
from math import sin, cos, radians, sqrt
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mne

OUT_DIR = Path("outputs")
LOCS_PATH = Path("data/channel_62_pos.locs")

BANDS = ["δ", "θ", "α", "β", "γ"]
CLASSES = ["Neutral", "Sad", "Fear", "Happy", "Disgust"]

HEAD_RADIUS = 0.095  # metres


def build_mne_info() -> mne.Info:
    ch_names, ch_pos = [], {}
    for line in LOCS_PATH.read_text().splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        angle = float(parts[1])
        radius = float(parts[2])
        name = parts[3]
        x = radius * sin(radians(angle)) * HEAD_RADIUS
        y = radius * cos(radians(angle)) * HEAD_RADIUS
        z = sqrt(max(0.0, 1.0 - radius**2)) * HEAD_RADIUS
        ch_names.append(name)
        ch_pos[name] = np.array([x, y, z])

    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame="head")
    info = mne.create_info(ch_names=ch_names, sfreq=1.0, ch_types="eeg")
    info.set_montage(montage)
    return info


def plot_subject(npz_path: Path, info: mne.Info, fig_dir: Path) -> None:
    data = np.load(npz_path)
    mean_de = data["mean_de"]  # (5, 62, 5): class, channel, band
    acc = float(data["acc"])
    f1 = float(data["f1"])

    fig, axes = plt.subplots(
        5,
        5,
        figsize=(16, 14),
        gridspec_kw={"hspace": 0.5, "wspace": 0.1},
    )
    fig.suptitle(f"{npz_path.stem}  acc={acc:.3f}  f1={f1:.3f}", fontsize=13)

    vmin = mean_de.min()
    vmax = mean_de.max()

    for cls_idx, cls_name in enumerate(CLASSES):
        for band_idx, band_name in enumerate(BANDS):
            ax = axes[cls_idx][band_idx]
            values = mean_de[cls_idx, :, band_idx]  # (62,)
            mne.viz.plot_topomap(
                values,
                info,
                axes=ax,
                show=False,
                contours=4,
                vlim=(vmin, vmax),
            )
            if cls_idx == 0:
                ax.set_title(band_name, fontsize=10)
            if band_idx == 0:
                ax.set_ylabel(
                    cls_name, fontsize=9, rotation=0, labelpad=40, va="center"
                )

    fig_dir.mkdir(parents=True, exist_ok=True)
    out = fig_dir / f"{npz_path.stem}_topomap.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["per-subject", "loocv"], default="per-subject"
    )
    parser.add_argument(
        "--subject", default="all", help="Subject stem (e.g. '3') or 'all'"
    )
    args = parser.parse_args()

    mne.set_log_level("WARNING")
    info = build_mne_info()

    mode_dir = args.mode.replace("-", "_")
    npz_dir = OUT_DIR / mode_dir
    if not npz_dir.exists():
        raise FileNotFoundError(f"No outputs found at {npz_dir}. Run main.py first.")

    fig_dir = OUT_DIR / "figures" / mode_dir

    if args.subject == "all":
        files = sorted(npz_dir.glob("*.npz"))
    else:
        stem = args.subject if args.mode == "per-subject" else f"fold_{args.subject}"
        files = [npz_dir / f"{stem}.npz"]

    if not files:
        raise FileNotFoundError(f"No .npz files found in {npz_dir}")

    for f in files:
        plot_subject(f, info, fig_dir)


if __name__ == "__main__":
    main()
