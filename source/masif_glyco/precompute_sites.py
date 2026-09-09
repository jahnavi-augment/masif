#!/usr/bin/env python
"""
precompute_sites.py: build MaSIF patches at the glycosites of one protein.

The attachment N atom (ND2) of each glycosite is matched to its nearest mesh
vertex, and only those vertices are passed down through read_data_from_surface()
into compute_polar_coordinates() as centers.

Usage:
    python precompute_sites.py P01234

Run from data/masif_glyco/.
"""

import argparse
import json
import os
import sys

import numpy as np
import pymesh
from Bio.PDB import PDBParser
from scipy.spatial import cKDTree

from masif_glyco import glyco_config as cfg
from masif_modules.read_data_from_surface import read_data_from_surface

ARRAY_NAMES = ("input_feat", "rho", "theta", "mask")


def sequence_window(residues, resnum):
    """+/- SEQ_WINDOW residues as one-letter codes, '-' outside the model."""
    return "".join(
        cfg.THREE_TO_ONE.get(residues[resnum + off].get_resname().strip(), "X")
        if resnum + off in residues
        else "-"
        for off in range(-cfg.SEQ_WINDOW, cfg.SEQ_WINDOW + 1)
    )


def attachment_sites(pdb_path, sites):
    """
    Locate the ND2 atom of each requested glycosite in the extracted chain.

    Yields (site, coord, plddt, seq_window). A site that is absent, is not an
    ASN, or has no ND2 atom is skipped -- a numbering or isoform mismatch.

    plddt is AlphaFold's per-residue confidence, taken from the B-factor column.
    It is recorded because these models keep their disordered linkers, where a
    patch describes a prediction artefact rather than real structure; without it
    there is no way to restrict later analysis to high-confidence sites.
    """
    structure = PDBParser(QUIET=True).get_structure("x", pdb_path)
    residues = {
        r.get_id()[1]: r
        for r in structure[0][cfg.CHAIN_ID]
        if r.get_id()[0] == " "
    }

    for site in sites:
        residue = residues.get(site["resnum"])
        if residue is None or residue.get_resname().strip() != cfg.N_LINKED_RESNAME:
            continue
        if cfg.N_LINKED_ATTACH_ATOM not in residue:
            continue
        atom = residue[cfg.N_LINKED_ATTACH_ATOM]
        yield (
            site,
            np.array(atom.get_coord(), dtype=float),
            float(atom.get_bfactor()),
            sequence_window(residues, site["resnum"]),
        )


def compute_patches(ply_path, params, centers):
    """
    Patches for every center, tolerating centers that cannot be parameterised --
    one sitting on a mesh boundary, or in a badly connected region. The batch is
    retried center by center so a single bad site does not lose the protein.

    Returns (arrays, kept), where kept indexes into centers.
    """
    geometry_failure = (ValueError, IndexError, KeyError)
    try:
        arrays = read_data_from_surface(ply_path, params, centers=centers)[:4]
        return arrays, list(range(len(centers)))
    except geometry_failure:
        pass

    per_center = []
    kept = []
    for j, vertex in enumerate(centers):
        try:
            per_center.append(read_data_from_surface(ply_path, params, centers=[vertex])[:4])
            kept.append(j)
        except geometry_failure:
            continue

    if not kept:
        raise ValueError("no glycosite could be parameterised")
    arrays = tuple(
        np.concatenate([patch[k] for patch in per_center], axis=0) for k in range(4)
    )
    return arrays, kept


def precompute(acc, sites):
    ply_path = cfg.ply_path(acc)
    pdb_path = cfg.chain_pdb_path(acc)
    for path in (ply_path, pdb_path):
        if not os.path.exists(path):
            raise IOError("missing {} (run 01-pdb_extract_and_triangulate)".format(path))

    mesh = pymesh.load_mesh(ply_path)
    tree = cKDTree(np.asarray(mesh.vertices))

    centers = []
    rows = []
    for site, coord, plddt, seq_window in attachment_sites(pdb_path, sites):
        d_min, vertex = tree.query(coord)
        rows.append(
            {
                "acc": acc,
                "resnum": site["resnum"],
                "label": site["label"],
                "split": site["split"],
                "row": len(centers),
                "center_vertex": int(vertex),
                "d_min": float(d_min),
                "n_surface_vertices_near": len(
                    tree.query_ball_point(coord, cfg.EXPOSURE_PROBE_RADIUS)
                ),
                "plddt": plddt,
                "seq_window": seq_window,
                "buried": bool(d_min > cfg.BURIED_SITE_CUTOFF),
            }
        )
        centers.append(int(vertex))

    if not centers:
        raise ValueError("no usable glycosites for {}".format(acc))

    params = {"max_distance": cfg.PATCH_RADIUS, "max_shape_size": cfg.MAX_SHAPE_SIZE}
    arrays, kept = compute_patches(ply_path, params, centers)

    kept_set = set(kept)
    no_patch = [row["resnum"] for j, row in enumerate(rows) if j not in kept_set]
    rows = [rows[j] for j in kept]
    for position, row in enumerate(rows):
        row["row"] = position

    out_dir = cfg.site_patch_dir(acc)
    cfg.ensure_dir(out_dir)
    for name, array in zip(ARRAY_NAMES, arrays):
        np.save(os.path.join(out_dir, name + ".npy"), array.astype(np.float32))
    with open(os.path.join(out_dir, "sites.json"), "w") as handle:
        json.dump({"acc": acc, "n_patches": len(rows), "sites": rows,
                   "dropped_no_patch": no_patch}, handle, indent=2)

    no_atom = len(sites) - len(centers)
    print("{}: {} patches from {} mesh vertices{}{}".format(
        acc, len(rows), len(mesh.vertices),
        "; {} sites not in structure".format(no_atom) if no_atom else "",
        "; {} sites unparameterisable {}".format(len(no_patch), no_patch) if no_patch else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accession")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_json = os.path.join(cfg.site_patch_dir(args.accession), "sites.json")
    if os.path.exists(out_json) and not args.force:
        print("{}: already precomputed".format(args.accession))
        return

    grouped = cfg.sites_by_protein()
    if args.accession not in grouped:
        sys.exit("{} is not in {}".format(args.accession, cfg.SITES_CSV))
    precompute(args.accession, grouped[args.accession])


if __name__ == "__main__":
    main()
