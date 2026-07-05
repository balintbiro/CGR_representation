# import the necessary libraries
import os
import click
import torch
import random
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import shap
from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from sklearn.model_selection import train_test_split

from utils import loggerConfig, Cnn

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


def per_image_shap(
    X: np.ndarray,
    y: np.ndarray,
    res: int,
    task: str,
    n_background: int,
    n_explain: int,
) -> dict:
    # seeding for reproducibility of the split, training and SHAP background sampling
    seed = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if task == "binary":
        y = y.astype("float32")
        net = NeuralNetBinaryClassifier(
            Cnn(output_dim=1),
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
        )
    else:
        y = y.astype(np.int64)
        net = NeuralNetClassifier(
            Cnn(output_dim=len(np.unique(y))),
            criterion=torch.nn.CrossEntropyLoss,
            max_epochs=10,
            lr=0.001,
            optimizer=torch.optim.Adam,
            device=device,
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )

    logger.info("Training the CNN on %d samples", X_train.shape[0])
    net.fit(X_train, y_train)

    # predictions on the held-out set define which maps are "right" vs "wrong"
    pred = net.predict(X_test)
    if task == "binary":
        pred = pred.astype("float32").ravel()
    else:
        pred = pred.astype(np.int64).ravel()
    correct = pred == y_test

    # optionally cap how many test images we attribute (SHAP is the slow part)
    if n_explain and n_explain > 0:
        n_explain = min(n_explain, X_test.shape[0])
    else:
        n_explain = X_test.shape[0]
    X_test = X_test[:n_explain]
    y_test = y_test[:n_explain]
    pred = pred[:n_explain]
    correct = correct[:n_explain]

    # the raw torch module is what SHAP back-propagates through
    model = net.module_.to(device).eval()
    idx = np.random.permutation(X_train.shape[0])[: min(n_background, X_train.shape[0])]
    background = torch.tensor(X_train[idx], dtype=torch.float32, device=device)
    to_explain = torch.tensor(X_test, dtype=torch.float32, device=device)

    logger.info(
        "Computing SHAP: %d background, %d test images", background.shape[0], to_explain.shape[0]
    )
    explainer = shap.DeepExplainer(model, background)
    shap_values = explainer.shap_values(to_explain, check_additivity=False)

    n = X_test.shape[0]
    maps = np.empty((n, res, res), dtype=np.float32)
    if task == "binary":
        # single output: one map per image, shape (n, 1, res, res)
        sv = np.asarray(shap_values).reshape(n, res, res)
        maps[:] = sv
    else:
        # multiclass: keep the map for the class each image was predicted as.
        # newer shap returns (n, 1, res, res, n_classes); older returns a list.
        if isinstance(shap_values, list):
            for i in range(n):
                maps[i] = np.asarray(shap_values[int(pred[i])])[i].reshape(res, res)
        else:
            sv = np.asarray(shap_values)  # (n, 1, res, res, n_classes)
            for i in range(n):
                maps[i] = sv[i, 0, :, :, int(pred[i])]

    return {
        "maps": maps,
        "images": X_test.reshape(n, res, res),
        "true_label": np.asarray(y_test),
        "pred_label": np.asarray(pred),
        "correct": np.asarray(correct),
    }


def _panel(ax, data, title):
    m = np.abs(data).mean(axis=0) if data.shape[0] else np.zeros_like(data[0:1])
    if data.shape[0] and m.max() > 0:
        m = m / m.max()
    im = ax.imshow(m, cmap="inferno", origin="lower")
    ax.set_title(title)
    ax.set_xlabel("FCGR x")
    ax.set_ylabel("FCGR y")
    return im


def save_correct_vs_wrong(result: dict, outfile: str, name: str) -> None:
    maps, correct = result["maps"], result["correct"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    n_c, n_w = int(correct.sum()), int((~correct).sum())
    im0 = _panel(axes[0], maps[correct], f"Correct (n={n_c}) — {name}")
    im1 = _panel(axes[1], maps[~correct], f"Wrong (n={n_w}) — {name}")
    fig.colorbar(im0, ax=axes[0], label="mean |SHAP| (norm.)")
    fig.colorbar(im1, ax=axes[1], label="mean |SHAP| (norm.)")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


def save_by_class(result: dict, outfile: str, name: str) -> None:
    maps, pred = result["maps"], result["pred_label"]
    classes = np.unique(pred)
    fig, axes = plt.subplots(1, len(classes), figsize=(4 * len(classes), 4), squeeze=False)
    for ax, c in zip(axes[0], classes):
        sel = pred == c
        im = _panel(ax, maps[sel], f"class {c} (n={int(sel.sum())})")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle(f"CNN focus per predicted class — {name}")
    fig.tight_layout()
    fig.savefig(outfile, dpi=200)
    plt.close(fig)


# define the command line interface using click
@click.command()
@click.option("--logfile", help="Path to logfile[.log]", required=True)
@click.option(
    "--fcgr_matrix",
    help="Path to file[.csv] containing FCGRs (same format as cnn_cv.py)",
    required=True,
)
@click.option("--outdir", help="Directory for the per-image maps and figures", required=True)
@click.option("--name", help="Dataset/encoding name, used for filenames and titles", required=True)
@click.option("--res", help="Resolution (int) of the FCGR", required=True, type=int)
@click.option(
    "--task",
    help="Classification task",
    type=click.Choice(["binary", "multiclass"], case_sensitive=False),
    default="binary",
    show_default=True,
)
@click.option(
    "--n-background",
    help="Number of training samples for the SHAP reference distribution",
    type=int,
    default=100,
    show_default=True,
)
@click.option(
    "--n-explain",
    help="Max number of test images to attribute (0 = all)",
    type=int,
    default=200,
    show_default=True,
)
def main(logfile, fcgr_matrix, outdir, name, res, task, n_background, n_explain) -> None:
    os.makedirs(outdir, exist_ok=True)
    loggerConfig(logfile=logfile)
    logger.info("Loading FCGR matrix from %s", fcgr_matrix)
    fcgr_df = pd.read_csv(fcgr_matrix)
    X = fcgr_df.drop(columns=["label"]).values.astype("float32").reshape(-1, 1, res, res)
    y = fcgr_df["label"].values

    result = per_image_shap(
        X=X,
        y=y,
        res=res,
        task=task.lower(),
        n_background=n_background,
        n_explain=n_explain,
    )

    # raw per-image maps + images, for arbitrary re-grouping later
    npz_path = os.path.join(outdir, f"{name}_shap_maps.npz")
    np.savez_compressed(
        npz_path,
        maps=result["maps"],
        images=result["images"],
        true_label=result["true_label"],
        pred_label=result["pred_label"],
        correct=result["correct"],
    )
    # a flat, human-readable index of every saved map
    meta = pd.DataFrame(
        {
            "map_index": np.arange(result["maps"].shape[0]),
            "true_label": result["true_label"],
            "pred_label": result["pred_label"],
            "correct": result["correct"],
        }
    )
    meta_path = os.path.join(outdir, f"{name}_metadata.csv")
    meta.to_csv(meta_path, index=False)

    # ready-made comparisons
    save_correct_vs_wrong(result, os.path.join(outdir, f"{name}_correct_vs_wrong.png"), name)
    if task.lower() == "multiclass":
        save_by_class(result, os.path.join(outdir, f"{name}_by_class.png"), name)

    acc = float(result["correct"].mean())
    logger.info(
        "Saved %d per-image maps to %s (test accuracy %.3f)",
        result["maps"].shape[0],
        npz_path,
        acc,
    )


if __name__ == "__main__":
    main()
