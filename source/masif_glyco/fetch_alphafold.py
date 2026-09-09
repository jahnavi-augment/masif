#!/usr/bin/env python
"""
fetch_alphafold.py: download AlphaFold DB models by UniProt accession.

Reads existing predictions only -- a mutated sequence would have to be folded
with AF2. AlphaFold residue numbering is UniProt numbering, so a glycosite
position indexes the structure directly.

Usage:
    python fetch_alphafold.py                 # every protein in lists/proteins.txt
    python fetch_alphafold.py P01234 P05678

Run from data/masif_glyco/.
"""

import argparse
import json
import os
import sys

from masif_glyco import glyco_config as cfg

try:
    from urllib.error import HTTPError, URLError
    from urllib.request import urlopen
except ImportError:  # python 2
    from urllib2 import HTTPError, URLError, urlopen

# The file URL is resolved through the API rather than constructed: AlphaFold DB
# bumps its model version and removes the old files, so a hardcoded
# AF-{acc}-F1-model_v4.pdb 404s for every accession once v5 lands.
AF_API_URL = "https://alphafold.ebi.ac.uk/api/prediction/{acc}"


def get(url, timeout=60):
    handle = urlopen(url, timeout=timeout)
    try:
        return handle.read()
    finally:
        handle.close()


def fetch(acc, force=False):
    """Download one model. Returns (ok, message)."""
    dest = cfg.raw_pdb_path(acc)
    if os.path.exists(dest) and not force:
        return True, "already present"
    cfg.ensure_dir(os.path.dirname(dest))

    try:
        records = json.loads(get(AF_API_URL.format(acc=acc)).decode("utf-8"))
        if not records:
            return False, "no AlphaFold entry"
        entry = records[0]
        if not entry.get("pdbUrl"):
            # Very large models are distributed as mmCIF only.
            return False, "no PDB file (mmCIF only)"
        pdb = get(entry["pdbUrl"])
    except HTTPError as exc:
        return False, "HTTP {}".format(exc.code)
    except (URLError, ValueError, IOError) as exc:
        return False, "{}".format(exc)

    with open(dest, "wb") as handle:
        handle.write(pdb)
    return True, "v{} model".format(entry.get("latestVersion", "?"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("accessions", nargs="*")
    ap.add_argument("--force", action="store_true", help="re-download existing files")
    args = ap.parse_args()

    accessions = args.accessions
    if not accessions:
        with open(cfg.PROTEINS_TXT) as handle:
            accessions = [line.strip() for line in handle if line.strip()]

    n_ok = 0
    for count, acc in enumerate(accessions, start=1):
        ok, message = fetch(acc, force=args.force)
        n_ok += ok
        print("[{}/{}] {} {}: {}".format(
            count, len(accessions), "ok  " if ok else "SKIP", acc, message))
        sys.stdout.flush()
    print("{}/{} models downloaded".format(n_ok, len(accessions)))


if __name__ == "__main__":
    main()
