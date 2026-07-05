# import the necessary libraries
import os
import click
import logging
import numpy as np
import matplotlib.pyplot as plt

from utils import loggerConfig

logger = logging.getLogger(__name__)


def load(npz_path: str) -> dict:
    d = np.load(npz_path)
    return {
        "maps": d["maps"],            # (n, res, res) signed SHAP, predicted class
        "images": d["images"],        # (n, res, res) input FCGRs
        "true_label": d["true_label"],
        "pred_label": d["pred_label"],
        "correct": d["correct"].astype(bool),
    }


def mean_focus(maps: np.ndarray) -> np.ndarray:
    if maps.shape[0] == 0:
        return None
    m = np.abs(maps).mean(axis=0)
    return m / m.max() if m.max() > 0 else m


def _heat(ax, data, title):
    if data is None:
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        return None
    im = ax.imshow(data, cmap="inferno", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("FCGR x")
    ax.set_ylabel("FCGR y")
    return im


def plot_correct_vs_wrong_by_class(data: dict, outfile: str, name: str) -> None:
    maps, pred, correct = data["maps"], data["pred_label"], data["correct"]
    classes = np.unique(pred)
    fig, axes = plt.subplots(
        len(classes), 2, figsize=(9, 4 * len(classes)), squeeze=False
    )
    for row, c in enumerate(classes):
        in_class = pred == c
        right = maps[in_class & correct]
        wrong = maps[in_class & ~correct]
        im0 = _heat(axes[row][0], mean_focus(right), f"class {c} correct (n={right.shape[0]})")
        im1 = _heat(axes[row][1], mean_focus(wrong), f"class {c} wrong (n={wrong.shape[0]})")
        for ax, im in ((axes[row][0], im0), (axes[row][1], im1)):
            if im is not None:
                fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"CNN focus — correct vs wrong per predicted class — {name}")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


def plot_overlay(data: dict, outfile: str, name: str) -> None:
    maps, images, correct = data["maps"], data["images"], data["correct"]
    groups = [("Correct", correct), ("Wrong", ~correct)]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, (label, sel) in zip(axes, groups):
        if sel.sum() == 0:
            _heat(ax, None, f"{label} (n=0)")
            continue
        bg = images[sel].mean(axis=0)
        bg = bg / bg.max() if bg.max() > 0 else bg
        focus = mean_focus(maps[sel])
        ax.imshow(bg, cmap="gray", origin="lower")
        im = ax.imshow(focus, cmap="inferno", origin="lower", alpha=0.55)
        ax.set_title(f"{label} (n={int(sel.sum())}) — {name}")
        ax.set_xlabel("FCGR x")
        ax.set_ylabel("FCGR y")
        fig.colorbar(im, ax=ax, fraction=0.046, label="mean |SHAP| (norm.)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)
    logger.info("Wrote %s", outfile)


# define the command line interface using click
@click.command()
@click.option("--logfile", help="Path to logfile[.log]", required=True)
@click.option(
    "--npz",
    help="Path to the *_shap_maps.npz produced by shap_explain.py",
    required=True,
)
@click.option("--outdir", help="Directory for the aggregate figures", required=True)
@click.option("--name", help="Dataset/encoding name for filenames and titles", required=True)
def main(logfile, npz, outdir, name) -> None:
    """
    Read a per-image SHAP stack and dump the standard cuts:
    correct-vs-wrong per class, and SHAP overlaid on the mean FCGR.
    """
    os.makedirs(outdir, exist_ok=True)
    loggerConfig(logfile=logfile)
    logger.info("Loading SHAP stack from %s", npz)
    data = load(npz)

    plot_correct_vs_wrong_by_class(
        data, os.path.join(outdir, f"{name}_correct_vs_wrong_by_class.png"), name
    )
    plot_overlay(data, os.path.join(outdir, f"{name}_overlay.png"), name)
    logger.info("Aggregate figures written to %s", outdir)


if __name__ == "__main__":
    main()
