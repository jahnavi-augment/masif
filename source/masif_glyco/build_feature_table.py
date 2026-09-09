#!/usr/bin/env python
"""
build_feature_table.py: join the per-protein descriptor files into flat
train/val/test tables.

Output (features/dataset.npz), per split:
    X_<split>           (n_sites, 80) float32
    y_<split>           (n_sites,) int
    groups_<split>      (n_sites,) accession, for grouped CV
    resnum_<split>      (n_sites,) int
    meta_<split>        (n_sites, 3) [d_min, n_surface_vertices_near, plddt]
    seq_window_<split>  (n_sites,) str

Usage:
    python build_feature_table.py

Run from data/masif_glyco/.
"""

import argparse
import json
import os
from collections import defaultdict

import numpy as np

from masif_glyco import glyco_config as cfg

META_NAMES = ["d_min", "n_surface_vertices_near", "plddt"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(cfg.FEATURE_DIR, "dataset.npz"))
    args = ap.parse_args()

    accessions = sorted({s["acc"] for s in cfg.read_sites()})
    data = defaultdict(lambda: defaultdict(list))
    n_proteins = 0

    for acc in accessions:
        if not os.path.exists(cfg.descriptor_path(acc)):
            continue
        with np.load(cfg.descriptor_path(acc), allow_pickle=False) as handle:
            desc = handle["desc"]
            manifest = json.loads(str(handle["sites_json"]))
        n_proteins += 1

        for site in manifest["sites"]:
            bundle = data[site["split"]]
            bundle["X"].append(desc[site["row"]])
            bundle["y"].append(site["label"])
            bundle["groups"].append(acc)
            bundle["resnum"].append(site["resnum"])
            bundle["meta"].append([site[name] for name in META_NAMES])
            bundle["seq_window"].append(site["seq_window"])

    if not data:
        raise SystemExit("no sites found; check descriptors/ and lists/sites.csv")

    out = {"meta_names": np.array(META_NAMES)}
    for split, bundle in sorted(data.items()):
        out["X_" + split] = np.asarray(bundle["X"], dtype=np.float32)
        out["y_" + split] = np.asarray(bundle["y"], dtype=np.int64)
        out["groups_" + split] = np.asarray(bundle["groups"])
        out["resnum_" + split] = np.asarray(bundle["resnum"], dtype=np.int64)
        out["meta_" + split] = np.asarray(bundle["meta"], dtype=np.float32)
        out["seq_window_" + split] = np.asarray(bundle["seq_window"])
        y = out["y_" + split]
        print("{:>5}: {:>6} sites  {:>5} proteins  {:.1%} positive".format(
            split, len(y), len(set(bundle["groups"])), y.mean()))

    cfg.ensure_dir(os.path.dirname(args.out))
    np.savez_compressed(args.out, **out)
    print("wrote {} from {} proteins".format(args.out, n_proteins))


if __name__ == "__main__":
    main()
