"""Parse the most recent log in logs/ and plot per-subject or per-fold training curves."""
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

LOG_DIR = Path("logs")
OUT_DIR = Path("outputs/figures")

# log line patterns
_LOOCV_RE = re.compile(
    r"LOOCV fold (\d+) epoch (\d+): "
    r"train_loss=([\d.]+) train_acc=([\d.]+) "
    r"val_loss=([\d.]+) val_acc=([\d.]+) val_f1=([\d.]+)"
)
_SUBJ_START_RE = re.compile(r"Start training (\S+)")
_SUBJ_EPOCH_RE = re.compile(
    r"Epoch (\d+): "
    r"train_loss=([\d.]+) train_acc=([\d.]+) "
    r"val_loss=([\d.]+) val_acc=([\d.]+) val_f1=([\d.]+)"
)


def parse_log(path: Path):
    """Return (mode, label→list of epoch dicts)."""
    records = defaultdict(list)   # label → [{epoch, train_loss, ...}, ...]
    mode = None
    current_subj = None

    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if m := _LOOCV_RE.search(line):
            mode = "loocv"
            fold, epoch = int(m[1]), int(m[2])
            records[f"fold {fold}"].append({
                "epoch": epoch,
                "train_loss": float(m[3]), "train_acc": float(m[4]),
                "val_loss": float(m[5]), "val_acc": float(m[6]), "val_f1": float(m[7]),
            })
        elif m := _SUBJ_START_RE.search(line):
            mode = mode or "per-subject"
            current_subj = Path(m[1]).stem
        elif m := _SUBJ_EPOCH_RE.search(line):
            mode = mode or "per-subject"
            label = current_subj or "?"
            records[label].append({
                "epoch": int(m[1]),
                "train_loss": float(m[2]), "train_acc": float(m[3]),
                "val_loss": float(m[4]), "val_acc": float(m[5]), "val_f1": float(m[6]),
            })

    return mode, records


def plot_curves(mode: str, records: dict, log_stem: str) -> None:
    labels = sorted(records.keys(), key=lambda s: int(re.search(r"\d+", s).group()))
    n = len(labels)
    cols = 4
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.35})
    axes_flat = axes.flat if rows > 1 else [axes] if cols == 1 else list(axes.flat)

    for ax, label in zip(axes_flat, labels):
        epochs_data = sorted(records[label], key=lambda d: d["epoch"])
        xs = [d["epoch"] for d in epochs_data]

        ax2 = ax.twinx()
        ax.plot(xs, [d["train_loss"] for d in epochs_data], "b-o", ms=4, label="train loss")
        ax.plot(xs, [d["val_loss"]   for d in epochs_data], "b--s", ms=4, label="val loss")
        ax2.plot(xs, [d["train_acc"] for d in epochs_data], "r-o", ms=4, label="train acc")
        ax2.plot(xs, [d["val_acc"]   for d in epochs_data], "r--s", ms=4, label="val acc")

        ax.set_title(label, fontsize=9)
        ax.set_xlabel("epoch", fontsize=7)
        ax.set_ylabel("loss", fontsize=7, color="b")
        ax2.set_ylabel("acc", fontsize=7, color="r")
        ax.tick_params(axis="y", labelcolor="b", labelsize=6)
        ax2.tick_params(axis="y", labelcolor="r", labelsize=6)
        ax.tick_params(axis="x", labelsize=6)

    # hide unused subplots
    for ax in list(axes_flat)[n:]:
        ax.set_visible(False)

    # shared legend on first axes
    handles = [
        plt.Line2D([0], [0], color="b", ls="-",  marker="o", ms=4, label="train loss"),
        plt.Line2D([0], [0], color="b", ls="--", marker="s", ms=4, label="val loss"),
        plt.Line2D([0], [0], color="r", ls="-",  marker="o", ms=4, label="train acc"),
        plt.Line2D([0], [0], color="r", ls="--", marker="s", ms=4, label="val acc"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(f"{log_stem}  ({mode})", fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out = OUT_DIR / f"training_{log_stem}.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def main():
    logs = sorted(LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if not logs:
        raise FileNotFoundError(f"No log files found in {LOG_DIR}")
    log_path = logs[-1]
    print(f"Parsing {log_path.name}")

    mode, records = parse_log(log_path)
    if not records:
        raise ValueError("No training metrics found in log.")

    plot_curves(mode, records, log_path.stem)


if __name__ == "__main__":
    main()
