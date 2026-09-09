#!/usr/bin/env python
"""
train_mlp.py: fit an MLP on the 80-d surface fingerprints.

Train and val are pooled and re-split into 10 folds grouped by accession, so
every site of a protein lands in the same fold and nothing leaks between train
and val. Test is held out throughout and scored once.

Usage:
    python train_mlp.py

Run from data/masif_glyco/.
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    matthews_corrcoef,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedGroupKFold

DATASET = "features/dataset.npz"
OUT_DIR = "models/mlp"

USE_WANDB = False
WANDB_PROJECT = "masif-glyco"
WANDB_ENTITY = None  # None uses the default entity
WANDB_TAGS = ["frozen-descriptors"]

HIDDEN = [128, 64]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 256
MAX_EPOCHS = 200
PATIENCE = 25
SEED = 10

# Early stopping on "loss" or "auprc"
STOP_ON = "loss"

N_FOLDS = 10

CONFIG = {
    "dataset": DATASET, "out_dir": OUT_DIR,
    "hidden": HIDDEN, "dropout": DROPOUT, "lr": LR,
    "weight_decay": WEIGHT_DECAY, "batch_size": BATCH_SIZE,
    "max_epochs": MAX_EPOCHS, "patience": PATIENCE, "seed": SEED,
    "n_folds": N_FOLDS, "stop_on": STOP_ON,
}

SWEEPABLE = ("hidden", "dropout", "lr", "weight_decay", "batch_size",
             "max_epochs", "patience", "seed", "n_folds")


def setup_wandb():
    """
    Start a wandb run and let a sweep override the constants above.

    Returns the run, or None when wandb is off or unavailable; every call site
    guards on that, so the script runs identically with wandb absent.

    A sweep agent passes the trial's values through wandb.config. They are
    written back over the module globals so the functions below keep reading
    plain constants rather than threading a config object through every
    signature.
    """
    if not (USE_WANDB or os.environ.get("WANDB_SWEEP_ID")):
        return None
    try:
        import wandb
    except ImportError:
        print("wandb not installed; continuing without logging")
        return None

    run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY,
                     tags=WANDB_TAGS, config=CONFIG)
    resolved = {k: v for k, v in dict(run.config).items() if k in SWEEPABLE}
    globals().update({k.upper(): v for k, v in resolved.items()})
    CONFIG.update(resolved)

    wandb.define_metric("epoch")
    wandb.define_metric("fold/*", step_metric="epoch")
    return run


def load_dataset(path):
    data = np.load(path, allow_pickle=False)
    splits = {}
    for key in data.files:
        if not key.startswith("X_"):
            continue
        split = key[2:]
        X = data[key]
        splits[split] = {
            "X": X.astype(np.float32),
            "y": data["y_" + split].astype(np.float32),
            "groups": data["groups_" + split],
        }
    for required in ("train", "val"):
        if required not in splits:
            raise SystemExit("{} has no {} split".format(path, required))
    return splits


class MLP(nn.Module):
    def __init__(self, n_in, hidden, dropout):
        super().__init__()
        layers = []
        for width in hidden:
            layers += [nn.Linear(n_in, width), nn.BatchNorm1d(width),
                       nn.ReLU(), nn.Dropout(dropout)]
            n_in = width
        layers.append(nn.Linear(n_in, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def predict(model, X, device):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.as_tensor(X, device=device))).cpu().numpy()


def standardizer(X):
    """Mean/std from these rows only, and the function that applies them."""
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1.0
    return mean, std, lambda A: ((A - mean) / std).astype(np.float32)


def epoch_logger(run, tag):
    """
    Per-epoch callback for train_one, or None when wandb is off.

    Each fold logs its own metric series against a shared `epoch` axis so the
    curves overlay, rather than concatenating into one sawtooth against a global
    step that restarts every fold.
    """
    if run is None:
        return None

    def log(epoch, train_loss, val_loss, val_auprc):
        record = {"epoch": epoch, "fold/{}/train_loss".format(tag): train_loss}
        # The final refit runs without a validation split, so these are absent
        # for that curve rather than zero.
        if val_loss is not None:
            record["fold/{}/val_loss".format(tag)] = val_loss
        if val_auprc is not None:
            record["fold/{}/val_auprc".format(tag)] = val_auprc
        run.log(record)

    return log


def train_one(Xtr, ytr, device, val=None, epochs=None, on_epoch=None):
    """
    Fit an MLP, early-stopping on val when one is given.

    Returns (model, sel_auprc, best_epoch), where sel_auprc is the AUPRC at the
    epoch STOP_ON selected -- not the best AUPRC reached, unless STOP_ON is
    "auprc". It is recorded so the per-fold figure stays comparable whichever
    criterion is in use.
    """
    Xtr_t = torch.as_tensor(Xtr, device=device)
    ytr_t = torch.as_tensor(ytr, device=device)

    model = MLP(Xtr.shape[1], HIDDEN, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    criterion = nn.BCEWithLogitsLoss()

    best_watched = float("-inf")
    sel_auprc, best_epoch, best_state, stale = -1.0, 0, None, 0
    for epoch in range(1, (epochs or MAX_EPOCHS) + 1):
        model.train()
        perm = torch.randperm(len(ytr_t), device=device)
        epoch_loss, n_batches = 0.0, 0
        for start in range(0, len(perm), BATCH_SIZE):
            batch = perm[start : start + BATCH_SIZE]
            if len(batch) < 2:  # BatchNorm needs at least two samples
                continue
            optimizer.zero_grad()
            loss = criterion(model(Xtr_t[batch]), ytr_t[batch])
            loss.backward()
            optimizer.step()
            epoch_loss += float(loss.detach())
            n_batches += 1
        train_loss = epoch_loss / max(n_batches, 1)

        auprc, val_loss = None, None
        if val is not None:
            model.eval()
            with torch.no_grad():
                logits = model(torch.as_tensor(val[0], device=device))
                targets = torch.as_tensor(val[1], device=device)
                val_loss = float(criterion(logits, targets))
                scores = torch.sigmoid(logits).cpu().numpy()
            auprc = average_precision_score(val[1], scores)
        if on_epoch is not None:
            on_epoch(epoch, train_loss, val_loss, auprc)
        if val is None:
            continue
        # Negated so "higher is better" holds for both criteria.
        watched = -val_loss if STOP_ON == "loss" else auprc
        if watched > best_watched:
            best_watched, sel_auprc, best_epoch, stale = watched, auprc, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, sel_auprc, best_epoch


def best_threshold(y, scores):
    """The threshold maximising MCC, over a quantile grid of the scores."""
    candidates = np.unique(np.quantile(scores, np.linspace(0.01, 0.99, 199)))
    mccs = [matthews_corrcoef(y, (scores >= t).astype(int)) for t in candidates]
    best = int(np.argmax(mccs))
    return float(candidates[best]), float(mccs[best])


def cross_validate(pooled, device, run=None):
    """
    Grouped k-fold over train+val, then a final model refit on all of it.

    Folds are grouped by accession so no protein straddles a boundary, and
    stratified so each keeps the same class balance as original dataset. The threshold comes from the
    pooled out-of-fold (oof) predictions rather than from any single fold. The final
    model runs for the mean stopping epoch without early stopping, there being no
    held-out data left to stop on.
    """
    print("{} sites over {} proteins pooled from train+val, {}-fold grouped CV".format(
        len(pooled["y"]), len(set(pooled["groups"])), N_FOLDS))

    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True,
                                    random_state=SEED)
    oof = np.zeros(len(pooled["y"]))
    fold_auprc, fold_epochs = [], []

    for fold, (tr, va) in enumerate(
        splitter.split(pooled["X"], pooled["y"], groups=pooled["groups"]), start=1
    ):
        _, _, normalize = standardizer(pooled["X"][tr])
        Xva = normalize(pooled["X"][va])
        model, sel_auprc, epoch = train_one(
            normalize(pooled["X"][tr]), pooled["y"][tr], device,
            val=(Xva, pooled["y"][va]),
            on_epoch=epoch_logger(run, fold),
        )
        oof[va] = predict(model, Xva, device)
        fold_auprc.append(sel_auprc)
        fold_epochs.append(epoch)
        print("  fold {:>2}: {:>5} held out, AUPRC@sel {:.4f}, epoch {}".format(
            fold, len(va), sel_auprc, epoch))
        if run is not None:
            run.log({"fold_summary/auprc": sel_auprc, "fold_summary/best_epoch": epoch,
                     "fold_summary/n_held_out": len(va), "fold_summary/fold": fold})

    threshold, oof_mcc = best_threshold(pooled["y"], oof)
    final_epochs = max(1, int(round(float(np.mean(fold_epochs)))))
    print("CV AUPRC@sel {:.4f} +/- {:.4f}; out-of-fold MCC {:.4f} at threshold {:.3f}".format(
        np.mean(fold_auprc), np.std(fold_auprc), oof_mcc, threshold))
    print("refitting on all {} sites for {} epochs".format(len(pooled["y"]), final_epochs))

    mean, std, normalize = standardizer(pooled["X"])
    model, _, _ = train_one(normalize(pooled["X"]), pooled["y"], device,
                            on_epoch=epoch_logger(run, "refit"),
                            epochs=final_epochs)
    return model, mean, std, threshold, {
        "folds": N_FOLDS,
        "fold_auprc": [float(a) for a in fold_auprc],
        "fold_epochs": fold_epochs,
        "cv_auprc_mean": float(np.mean(fold_auprc)),
        "cv_auprc_std": float(np.std(fold_auprc)),
        "oof_mcc": oof_mcc,
        "threshold": threshold,
        "final_epochs": final_epochs,
    }


def main():
    run = setup_wandb()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    splits = load_dataset(DATASET)
    pooled = {
        key: np.concatenate([splits["train"][key], splits["val"][key]])
        for key in ("X", "y", "groups")
    }
    model, mean, std, threshold, report = cross_validate(pooled, device, run)
    report["config"] = CONFIG

    if "test" in splits:
        yte = splits["test"]["y"]
        scores = predict(model, ((splits["test"]["X"] - mean) / std).astype(np.float32),
                         device)
        predicted = (scores >= threshold).astype(int)
        report["test"] = {
            "n": len(yte),
            "positive_rate": float(yte.mean()),
            "auprc": float(average_precision_score(yte, scores)),
            "roc_auc": float(roc_auc_score(yte, scores)),
            "mcc": float(matthews_corrcoef(yte, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(yte, predicted)),
        }
        print("\nTEST n={n}  positive_rate={positive_rate:.1%}\n"
              "  AUPRC    {auprc:.4f}\n"
              "  ROC-AUC  {roc_auc:.4f}\n"
              "  MCC      {mcc:.4f}\n"
              "  bal.acc  {balanced_accuracy:.4f}".format(**report["test"]))

    os.makedirs(OUT_DIR, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std,
                "config": CONFIG}, os.path.join(OUT_DIR, "model.pt"))
    with open(os.path.join(OUT_DIR, "report.json"), "w") as handle:
        json.dump(report, handle, indent=2)
    print("wrote {}/model.pt and report.json".format(OUT_DIR))

    if run is not None:
        run.summary.update({
            "cv_auprc_mean": report["cv_auprc_mean"],
            "cv_auprc_std": report["cv_auprc_std"],
            "oof_mcc": report["oof_mcc"],
            "threshold": report["threshold"],
            "final_epochs": report["final_epochs"],
        })
        if "test" in report:
            run.summary.update({"test/" + k: v for k, v in report["test"].items()})
        run.finish()


if __name__ == "__main__":
    main()
