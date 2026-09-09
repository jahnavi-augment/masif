"""
compute_site_descriptors.py: run the frozen pretrained MaSIF-search network over
the glycosite patches and save the 80-d fingerprints.

Usage:
    python compute_site_descriptors.py            # all of lists/proteins.txt
    python compute_site_descriptors.py P01234

Run from data/masif_glyco/.
"""

import argparse
import json
import os
import sys

import numpy as np
from masif_modules.MaSIF_ppi_search import MaSIF_ppi_search
from masif_modules.train_ppi_search import compute_val_test_desc

from masif_glyco import glyco_config as cfg


def build_network():
    learning_obj = MaSIF_ppi_search(
        cfg.PATCH_RADIUS,
        n_thetas=16,
        n_rhos=5,
        n_rotations=16,
        idx_gpu="/cpu:0",
        feat_mask=cfg.FEAT_MASK,
    )
    checkpoint = os.path.join(cfg.PRETRAINED_MODEL_DIR, "model")
    if not os.path.exists(checkpoint + ".index"):
        sys.exit(f"pretrained weights not found at {checkpoint}.index")
    learning_obj.saver.restore(learning_obj.session, checkpoint)
    print(f"restored frozen MaSIF-search weights from {checkpoint}")
    return learning_obj


def descriptors_for(learning_obj, acc, batch_size):
    patch_dir = cfg.site_patch_dir(acc)
    with open(os.path.join(patch_dir, "sites.json")) as handle:
        manifest = json.load(handle)
    load = lambda name: np.load(os.path.join(patch_dir, name + ".npy"))

    input_feat = load("input_feat")
    # Drop the feature channels feat_mask zeroes out.
    input_feat = np.delete(
        input_feat, np.where(np.array(cfg.FEAT_MASK) == 0.0)[0], axis=2
    )

    rho = load("rho")
    desc = compute_val_test_desc(
        learning_obj=learning_obj,
        idx=np.arange(len(rho)),
        rho_wrt_center=rho,
        theta_wrt_center=load("theta"),
        input_feat=input_feat,
        mask=load("mask"),
        batch_size=batch_size,
        flip=False,
    )
    return manifest, np.atleast_2d(desc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accessions", nargs="*")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--batch-size", type=int, default=1000)
    args = ap.parse_args()

    accessions = args.accessions
    if not accessions:
        with open(cfg.PROTEINS_TXT) as handle:
            accessions = [line.strip() for line in handle if line.strip()]

    todo = [
        acc
        for acc in accessions
        if os.path.exists(os.path.join(cfg.site_patch_dir(acc), "sites.json"))
        and (args.force or not os.path.exists(cfg.descriptor_path(acc)))
    ]
    if not todo:
        print(f"nothing to do ({len(accessions)} accessions given)")
        return

    print(
        f"computing descriptors for {len(todo)} of {len(accessions)} accessions"
    )
    cfg.ensure_dir(cfg.DESCRIPTOR_DIR)
    learning_obj = build_network()

    failures = []
    for count, acc in enumerate(todo, start=1):
        manifest, desc = descriptors_for(learning_obj, acc, args.batch_size)
        np.savez_compressed(
            cfg.descriptor_path(acc),
            desc=desc.astype(np.float32),
            sites_json=json.dumps(manifest),
        )
        print(f"[{count}/{len(todo)}] {acc}: {len(desc)} patches")

    if failures:
        cfg.ensure_dir(cfg.LOG_DIR)
        path = os.path.join(cfg.LOG_DIR, "descriptor_failures.txt")
        with open(path, "w") as handle:
            handle.write("".join(f"{a}\t{m}\n" for a, m in failures))
        print(f"{len(failures)} failures -> {path}")
    print(f"{len(todo) - len(failures)}/{len(todo)} descriptor files written")


if __name__ == "__main__":
    main()
