"""
glyco_config.py: shared paths and constants for the N-glycosylation application.

Relative paths resolve against the working directory, which every MaSIF
application expects to be its own data/<app>/ directory. MASIF_DATA_ROOT
prefixes them instead, matching default_config.masif_opts.
"""

import csv
import os
from collections import defaultdict

from default_config.masif_opts import masif_opts

# Fixed by the pretrained MaSIF-search weights: max_rho is baked into the
# Gaussian grid and sigma_rho_init = max_rho/8, so changing these invalidates
# the weights.
PATCH_RADIUS = 12.0
MAX_SHAPE_SIZE = 200
FEAT_MASK = [1.0] * 5

# The pretrained MaSIF-search network, reused frozen as a feature extractor.
PRETRAINED_MODEL_DIR = os.environ.get(
    "MASIF_GLYCO_MODEL_DIR", "../masif_ppi_search/nn_models/sc05/all_feat/model_data/"
)

CHAIN_ID = "A"

_DATA_ROOT = os.environ.get("MASIF_DATA_ROOT", "")


def _data_path(relative):
    return os.path.join(_DATA_ROOT, relative) if _DATA_ROOT else relative


SITE_PATCH_DIR = _data_path("data_preparation/05-site_patches/")
DESCRIPTOR_DIR = _data_path("descriptors/")
LIST_DIR = _data_path("lists/")
FEATURE_DIR = _data_path("features/")
LOG_DIR = _data_path("logs/")

SITES_CSV = os.path.join(LIST_DIR, "sites.csv")
PROTEINS_TXT = os.path.join(LIST_DIR, "proteins.txt")
SITES_COLUMNS = ["acc", "resnum", "label", "split"]

N_LINKED_ATTACH_ATOM = "ND2"
N_LINKED_RESNAME = "ASN"

BURIED_SITE_CUTOFF = 5.0

EXPOSURE_PROBE_RADIUS = 5.0

SEQ_WINDOW = 7

THREE_TO_ONE = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C",
    "GLN": "Q", "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I",
    "LEU": "L", "LYS": "K", "MET": "M", "PHE": "F", "PRO": "P",
    "SER": "S", "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
    "SEC": "U", "PYL": "O", "MSE": "M",
}


def raw_pdb_path(acc):
    return os.path.join(masif_opts["raw_pdb_dir"], acc + ".pdb")


def chain_pdb_path(acc):
    return os.path.join(masif_opts["pdb_chain_dir"], "{}_{}.pdb".format(acc, CHAIN_ID))


def ply_path(acc):
    return masif_opts["ply_file_template"].format(acc, CHAIN_ID)


def site_patch_dir(acc):
    return os.path.join(SITE_PATCH_DIR, acc)


def descriptor_path(acc):
    return os.path.join(DESCRIPTOR_DIR, acc + ".npz")


def ensure_dir(path):
    if path and not os.path.exists(path):
        os.makedirs(path)


def read_sites(path=None):
    """lists/sites.csv as a list of dicts with typed fields."""
    with open(path or SITES_CSV) as handle:
        return [
            {
                "acc": row["acc"].strip(),
                "resnum": int(row["resnum"]),
                "label": int(row["label"]),
                "split": row["split"].strip(),
            }
            for row in csv.DictReader(handle)
        ]


def sites_by_protein(path=None):
    """read_sites() grouped by accession, each list sorted by resnum."""
    grouped = defaultdict(list)
    for site in read_sites(path):
        grouped[site["acc"]].append(site)
    for group in grouped.values():
        group.sort(key=lambda s: s["resnum"])
    return dict(grouped)
