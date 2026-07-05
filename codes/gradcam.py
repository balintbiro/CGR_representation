# import the necessary libraries
import os
import click
import torch
import random
import logging
import numpy as np
import pandas as pd
import torch.nn.functional as F
import matplotlib.pyplot as plt

from skorch import NeuralNetBinaryClassifier, NeuralNetClassifier
from sklearn.model_selection import train_test_split

from utils import loggerConfig, Cnn

device = "cuda" if torch.cuda.is_available() else "cpu"

logger = logging.getLogger(__name__)


def _gradcam_batch(model, target_layer, x, pred, task):
    activations, gradients = {}, {}

    def fwd_hook(_, __, output):
        activations["value"] = output

    def bwd_hook(_, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    try:
        model.zero_grad()
        out = model(x)
        if task == "binary":
            # single logit: use +logit where the call was positive, -logit where
            # negative, so the CAM always explains the decision that was made
            logit = out.reshape(-1)
            sign = torch.tensor(
                np.where(pred > 0.5, 1.0, -1.0), dtype=torch.float32, device=x.device
            )
            selected = (sign * logit).sum()
        else:
            rows = torch.arange(out.shape[0], device=x.device)
            selected = out[rows, torch.tensor(pred, dtype=torch.long, device=x.device)].sum()
        selected.backward()

        A = activations["value"]            # (b, ch, h, w)
        grad = gradients["value"]           # (b, ch, h, w)
        weights = grad.mean(dim=(2, 3), keepdim=True)   # (b, ch, 1, 1)
        cam = F.relu((weights * A).sum(dim=1))          # (b, h, w)
        return cam.detach().cpu().numpy()
    finally:
        h1.remove()
        h2.remove()


def per_image_gradcam(
    X: np.ndarray,
    y: np.ndarray,
    res: int,
    task: str,
    n_explain: int,
    batch_size: int = 256,
) -> dict:
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

    pred = net.predict(X_test)
    if task == "binary":
        pred = pred.astype("float32").ravel()
    else:
        pred = pred.astype(np.int64).ravel()
    correct = pred == y_test

    if n_explain and n_explain > 0:
        n_explain = min(n_explain, X_test.shape[0])
    else:
        n_explain = X_test.shape[0]
    X_test = X_test[:n_explain]
    y_test = y_test[:n_explain]
    pred = pred[:n_explain]
    correct = correct[:n_explain]

    model = net.module_.to(device).eval()
    target_layer = model.conv  # the only conv layer in Cnn

    logger.info("Computing Grad-CAM for %d test images", X_test.shape[0])
    maps = np.empty((X_test.shape[0], res, res), dtype=np.float32)
    for start in range(0, X_test.shape[0], batch_size):
        end = start + batch_size
        x = torch.tensor(X_test[start:end], dtype=torch.float32, device=device)
        cam = _gradcam_batch(model, target_layer, x, pred[start:end], task)  # (b, h, w)
        # upsample the coarse conv-resolution CAM back to the FCGR grid
        cam_t = torch.tensor(cam, dtype=torch.float32).unsqueeze(1)          # (b, 1, h, w)
        cam_t = F.interpolate(cam_t, size=(res, res), mode="bilinear", align_corners=False)
        cam_up = cam_t.squeeze(1).numpy()
        # per-image min-max normalisation to [0, 1]
        flat = cam_up.reshape(cam_up.shape[0], -1)
        lo = flat.min(axis=1, keepdims=True)
        hi = flat.max(axis=1, keepdims=True)
        denom = np.where(hi - lo > 0, hi - lo, 1.0)
        maps[start:end] = ((flat - lo) / denom).reshape(cam_up.shape)

    return {
        "maps": maps,
        "images": X_test.reshape(X_test.shape[0], res, res),
        "true_label": np.asarray(y_test),
        "pred_label": np.asarray(pred),
        "correct": np.asarray(correct),
    }


def _panel(ax, data, title):
    m = data.mean(axis=0) if data.shape[0] else np.zeros((1, 1))
    if data.shape[0] and m.max() > 0:
        m = m / m.max()
    im = ax.imshow(m, cmap="jet", origin="lower")
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
    fig.colorbar(im0, ax=axes[0], label="mean Grad-CAM (norm.)")
    fig.colorbar(im1, ax=axes[1], label="mean Grad-CAM (norm.)")
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
    fig.suptitle(f"CNN Grad-CAM per predicted class — {name}")
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
    "--n-explain",
    help="Max number of test images to map (0 = all)",
    type=int,
    default=0,
    show_default=True,
)
def main(logfile, fcgr_matrix, outdir, name, res, task, n_explain) -> None:
    os.makedirs(outdir, exist_ok=True)
    loggerConfig(logfile=logfile)
    logger.info("Loading FCGR matrix from %s", fcgr_matrix)
    fcgr_df = pd.read_csv(fcgr_matrix)
    X = fcgr_df.drop(columns=["label"]).values.astype("float32").reshape(-1, 1, res, res)
    y = fcgr_df["label"].values

    result = per_image_gradcam(
        X=X, y=y, res=res, task=task.lower(), n_explain=n_explain
    )

    npz_path = os.path.join(outdir, f"{name}_gradcam_maps.npz")
    np.savez_compressed(
        npz_path,
        maps=result["maps"],
        images=result["images"],
        true_label=result["true_label"],
        pred_label=result["pred_label"],
        correct=result["correct"],
    )
    meta = pd.DataFrame(
        {
            "map_index": np.arange(result["maps"].shape[0]),
            "true_label": result["true_label"],
            "pred_label": result["pred_label"],
            "correct": result["correct"],
        }
    )
    meta.to_csv(os.path.join(outdir, f"{name}_gradcam_metadata.csv"), index=False)

    save_correct_vs_wrong(result, os.path.join(outdir, f"{name}_gradcam_correct_vs_wrong.png"), name)
    if task.lower() == "multiclass":
        save_by_class(result, os.path.join(outdir, f"{name}_gradcam_by_class.png"), name)

    acc = float(result["correct"].mean())
    logger.info(
        "Saved %d Grad-CAM maps to %s (test accuracy %.3f)",
        result["maps"].shape[0],
        npz_path,
        acc,
    )


if __name__ == "__main__":
    main()
