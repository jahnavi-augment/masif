# MaSIF-glyco: N-glycosylation site prediction from surface fingerprints

Predicts whether a candidate N-glycosylation site is glycosylated, using the
local molecular surface around the site as the only structural input.

The pretrained MaSIF-search network is reused **frozen** as a feature extractor:
each glycosite becomes an 80-dimensional surface fingerprint, and a small MLP is
trained on those vectors. No MaSIF weights are retrained.

## Pipeline

```
occupancy/atlas_strict/{train,val,test}.csv     (uniprot, pos, label)
  |  prepare_dataset.py
  v
lists/sites.csv, lists/proteins.txt
  |  fetch_alphafold.py                  AlphaFold DB model, saved as-is
  |  01-pdb_extract_and_triangulate.py   reduce -> MSMS -> APBS -> .ply  [stock MaSIF]
  |  precompute_sites.py                 ND2 -> nearest vertex, 12 A geodesic patch
  v
data_preparation/05-site_patches/<ACC>/  (n_sites, 200, 5) features + polar coords
  |  compute_site_descriptors.py         frozen pretrained MaSIF-search
  v
descriptors/<ACC>.npz                    (n_sites, 80)
  |  build_feature_table.py
  v
features/dataset.npz
  |  train_mlp.py
  v
models/
```
## Running it

From this directory:

```bash
# 1. Manifest, locally.
python $MASIF/source/masif_glyco/prepare_dataset.py \
    ../../occupancy/atlas_strict/{train,val,test}.csv

# 2. Surfaces, then patches. Start with a --limit to check timing and cost.
modal run modal_run.py::prepare --limit 20
modal run modal_run.py::prepare
modal run modal_run.py::patches

# 3. Frozen network + feature table, then bring the table home.
modal run modal_run.py::descriptors
modal run modal_run.py::fetch_features

# 4. Model, locally.
python $MASIF/source/masif_glyco/train_mlp.py
```

`modal run modal_run.py::status` shows how far along a run is. Every step is
idempotent: a protein whose output already exists is skipped unless `--force`.

## How the MLP is validated

`train_mlp.py` **pools train and val and re-splits them into 10 folds grouped by
accession** — every site of a protein stays in one fold, and folds are stratified
to keep the class balance. 

## Key parameters

Fixed by the pretrained weights, in `source/masif_glyco/glyco_config.py`:

- `PATCH_RADIUS = 12.0` — geodesic radius. **Not adjustable.** `max_rho` sets the
  Gaussian grid and `sigma_rho_init = max_rho/8`; a 20 A patch would put its
  outer ring outside every Gaussian's support.
- `MAX_SHAPE_SIZE = 200` — vertices per patch.
- `FEAT_MASK = [1,1,1,1,1]` — `[shape_index, ddc, hbond, charge, hydrophobicity]`.

Ours: `BURIED_SITE_CUTOFF = 5.0` (a site whose nearest vertex is farther than
this is flagged `buried` in `sites.json`), `EXPOSURE_PROBE_RADIUS = 5.0`, and
`SEQ_WINDOW = 7`.

