#!/usr/bin/env python
"""
prepare_dataset.py: turn the occupancy CSVs into the manifest the rest of the
pipeline reads.

Reads the atlas_strict train/val/test CSVs, checks that sequence[pos] is an
asparagine on every row, and writes:

    lists/sites.csv     acc,resnum,label,split
    lists/proteins.txt  one UniProt accession per line

Positions in the source CSVs are 0-based; UniProt and AlphaFold number from 1,
so every position is written +1. The asparagine check is what verifies that
reading, and fails loudly if the upstream convention changes.

Usage:
    python prepare_dataset.py ../../occupancy/atlas_strict/{train,val,test}.csv

Run from data/masif_glyco/.
"""

import argparse
import csv
import os
import sys
from collections import Counter

from masif_glyco import glyco_config as cfg


def split_from_filename(path):
    stem = os.path.splitext(os.path.basename(path))[0].lower()
    for split in ("train", "val", "test"):
        if stem.startswith(split):
            return split
    sys.exit("{}: filename is not train/val/test".format(path))


def read_csv(path, split):
    with open(path) as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["split"] = split
    return rows


def verify_asn(rows):
    """The residue at each position must be an asparagine in the sequence."""
    bad = [r for r in rows if str(r["sequence"])[int(r["pos"])] != "N"]
    if bad:
        sys.exit(
            "{} of {} rows are not ASN at sequence[pos] -- positions are not "
            "0-based. First: {} pos {}".format(
                len(bad), len(rows), bad[0]["uniprot"], bad[0]["pos"]
            )
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv", nargs="+", help="train.csv val.csv test.csv")
    ap.add_argument("--out-dir", default=cfg.LIST_DIR)
    args = ap.parse_args()

    rows = []
    for path in args.input_csv:
        rows.extend(read_csv(path, split_from_filename(path)))
    print("read {} rows from {} file(s)".format(len(rows), len(args.input_csv)))
    verify_asn(rows)

    sites = []
    seen = set()
    for row in rows:
        acc = row["uniprot"].strip()
        resnum = int(row["pos"]) + 1
        # Isoform accessions have no AlphaFold DB model (it is canonical-only)
        # and their numbering may not be canonical either.
        if "-" in acc or (acc, resnum) in seen:
            continue
        seen.add((acc, resnum))
        sites.append(
            {
                "acc": acc,
                "resnum": resnum,
                "label": int(row["label"]),
                "split": row["split"],
            }
        )
    sites.sort(key=lambda s: (s["acc"], s["resnum"]))
    accessions = sorted({s["acc"] for s in sites})

    cfg.ensure_dir(args.out_dir)
    with open(os.path.join(args.out_dir, "sites.csv"), "w") as handle:
        writer = csv.DictWriter(handle, fieldnames=cfg.SITES_COLUMNS)
        writer.writeheader()
        writer.writerows(sites)
    with open(os.path.join(args.out_dir, "proteins.txt"), "w") as handle:
        handle.write("".join(acc + "\n" for acc in accessions))

    print("wrote {} sites over {} proteins to {}".format(
        len(sites), len(accessions), args.out_dir))
    counts = Counter((s["split"], s["label"]) for s in sites)
    for split in ("train", "val", "test"):
        pos, neg = counts.get((split, 1), 0), counts.get((split, 0), 0)
        if pos + neg:
            print("  {:>5}: {:>6} pos  {:>6} neg  ({:.1%} positive)".format(
                split, pos, neg, pos / float(pos + neg)))


if __name__ == "__main__":
    main()
