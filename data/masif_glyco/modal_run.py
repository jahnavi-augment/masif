"""
modal_run.py: run the MaSIF-glyco preprocessing on Modal.

Layout on the volume:

    /data/lists/sites.csv                                     uploaded from here
    /data/data_preparation/01-benchmark_surfaces/<ACC>_A.ply
    /data/data_preparation/05-site_patches/<ACC>/
    /data/descriptors/<ACC>.npz
    /data/features/dataset.npz                                <- what you want

Usage:
    modal run modal_run.py::prepare --limit 20    # phase A: surfaces
    modal run modal_run.py::patches               # phase B: glycosite patches
    modal run modal_run.py::descriptors           # frozen network + feature table
    modal run modal_run.py::fetch_features        # download dataset.npz
    modal run modal_run.py::status
"""

import os
import pathlib
import subprocess
import sys
import time

import modal

if modal.is_local():
    REPO = pathlib.Path(__file__).resolve().parents[2]
else:
    REPO = pathlib.Path("/masif")

PY36 = "/usr/local/bin/python3.6"
SOURCE = "/masif/source"
WEIGHTS = "/masif/weights/model_data"
DATA = "/data"
WORKDIR = "/root/work"
RAW_PDB_DIR = "/root/work/raw_pdbs"

BATCH_TIMEOUT = 4 * 60 * 60
STEP01_TIMEOUT = 60 * 60

app = modal.App("masif-glyco")
volume = modal.Volume.from_name("masif-glyco", create_if_missing=True)

image = (
    modal.Image.from_dockerfile(
        REPO / "data/masif_glyco/docker/Dockerfile",
        context_dir=REPO / "data/masif_glyco/docker",
        add_python="3.11",
    )
    .env(
        {
            "PYTHONPATH": SOURCE,
            "MASIF_COMPUTE_IFACE": "0",
            "MASIF_GLYCO_MODEL_DIR": WEIGHTS,
            "MASIF_DATA_ROOT": DATA,
            "MASIF_RAW_PDB_DIR": RAW_PDB_DIR,
        }
    )
    .add_local_dir(str(REPO / "source"), SOURCE, copy=True)
    .add_local_dir(
        str(REPO / "data/masif_ppi_search/nn_models/sc05/all_feat/model_data"),
        WEIGHTS,
        copy=True,
    )
)


def ply_path(acc):
    return os.path.join(DATA, "data_preparation/01-benchmark_surfaces",
                        "{}_A.ply".format(acc))


def patch_json_path(acc):
    return os.path.join(DATA, "data_preparation/05-site_patches", acc, "sites.json")


def run_step(cmd, timeout):
    """Run one MaSIF step. Returns (ok, tail of output)."""
    import signal

    os.makedirs(WORKDIR, exist_ok=True)
    proc = subprocess.Popen(cmd, cwd=WORKDIR, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True)

    def kill_tree():
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except OSError:
            pass

    try:
        try:
            out = proc.communicate(timeout=timeout)[0]
        except subprocess.TimeoutExpired:
            kill_tree()
            proc.communicate()
            return False, "TIMEOUT after {}s".format(timeout)
        return proc.returncode == 0, out.decode("utf-8", "replace")[-1500:]
    finally:
        if proc.poll() is None:
            kill_tree()


BATCH_KWARGS = dict(
    image=image,
    volumes={DATA: volume},
    timeout=BATCH_TIMEOUT,
    cpu=2.0,
    max_containers=100,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0, initial_delay=10.0),
)


@app.function(**BATCH_KWARGS)
def prepare_batch(accessions, force=False):
    """Phase A: fetch from AlphaFold DB and triangulate, one .ply per protein."""
    volume.reload()
    results = []

    for acc in accessions:
        if os.path.exists(ply_path(acc)) and not force:
            results.append({"acc": acc, "status": "cached"})
            continue

        record = {"acc": acc}
        tic = time.time()
        run_step([PY36, SOURCE + "/masif_glyco/fetch_alphafold.py", acc], timeout=600)
        raw_pdb = os.path.join(RAW_PDB_DIR, acc + ".pdb")
        if not os.path.exists(raw_pdb):
            record["status"] = "no_structure"
        else:
            ok, out = run_step(
                [PY36, SOURCE + "/data_preparation/01-pdb_extract_and_triangulate.py",
                 "{}_A".format(acc)],
                timeout=STEP01_TIMEOUT,
            )
            os.remove(raw_pdb)
            if ok and os.path.exists(ply_path(acc)):
                record["status"] = "ok"
            else:
                record.update(status="step01_failed", log=out)
        record["seconds"] = round(time.time() - tic, 1)
        results.append(record)
        volume.commit()

    return results


@app.function(**BATCH_KWARGS)
def patch_batch(accessions, force=False):
    """Phase B: cut the glycosite patches out of the surfaces from phase A."""
    volume.reload()
    results = []

    for acc in accessions:
        if os.path.exists(patch_json_path(acc)) and not force:
            results.append({"acc": acc, "status": "cached"})
            continue
        if not os.path.exists(ply_path(acc)):
            results.append({"acc": acc, "status": "no_surface"})
            continue

        tic = time.time()
        cmd = [PY36, SOURCE + "/masif_glyco/precompute_sites.py", acc]
        if force:
            cmd.append("--force")
        ok, out = run_step(cmd, timeout=1800)
        record = {"acc": acc, "seconds": round(time.time() - tic, 1),
                  "status": "ok" if ok else "precompute_failed"}
        if not ok:
            record["log"] = out
        results.append(record)
        volume.commit()

    return results


@app.function(image=image, volumes={DATA: volume}, timeout=4 * 60 * 60, cpu=4.0)
def descriptors_pass(force=False):
    """Frozen network over every patch, then the feature table."""
    volume.reload()
    cmd = [PY36, SOURCE + "/masif_glyco/compute_site_descriptors.py"]
    if force:
        cmd.append("--force")
    desc_ok, desc_log = run_step(cmd, timeout=4 * 60 * 60 - 600)
    table_ok, table_log = run_step(
        [PY36, SOURCE + "/masif_glyco/build_feature_table.py"], timeout=1800
    )
    volume.commit()
    return {"descriptors_ok": desc_ok, "descriptors_log": desc_log,
            "table_ok": table_ok, "table_log": table_log}


@app.function(image=image, volumes={DATA: volume}, timeout=1800)
def upload_lists(sites_csv, proteins_txt):
    """Seed the volume with the manifest prepare_dataset.py produced locally."""
    lists = os.path.join(DATA, "lists")
    os.makedirs(lists, exist_ok=True)
    for name, text in (("sites.csv", sites_csv), ("proteins.txt", proteins_txt)):
        with open(os.path.join(lists, name), "w") as handle:
            handle.write(text)
    volume.commit()


@app.function(image=image, volumes={DATA: volume}, timeout=1800)
def volume_status():
    volume.reload()

    def count(rel, suffix=""):
        path = os.path.join(DATA, rel)
        if not os.path.isdir(path):
            return 0
        return sum(1 for name in os.listdir(path) if name.endswith(suffix))

    return {
        "surfaces": count("data_preparation/01-benchmark_surfaces", ".ply"),
        "site_patches": count("data_preparation/05-site_patches"),
        "descriptors": count("descriptors", ".npz"),
        "has_feature_table": os.path.exists(os.path.join(DATA, "features/dataset.npz")),
    }


@app.function(image=image, volumes={DATA: volume}, timeout=1800)
def read_outputs():
    """Return the small final artefacts so the local side can write them out."""
    volume.reload()
    out = {}
    for rel in ("features/dataset.npz", "logs/descriptor_failures.txt"):
        path = os.path.join(DATA, rel)
        if os.path.exists(path):
            with open(path, "rb") as handle:
                out[rel] = handle.read()
    return out


# --------------------------------------------------------------------------
# Local entrypoints
# --------------------------------------------------------------------------

HERE = pathlib.Path(__file__).resolve().parent


def accessions(limit=0):
    path = HERE / "lists/proteins.txt"
    if not path.exists():
        sys.exit("lists/proteins.txt not found -- run prepare_dataset.py first")
    accs = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    return accs[:limit] if limit else accs


def run_phase(fn, accs, batch_size, force, label):
    from collections import Counter

    # One function call per batch, so a container handles several proteins.
    batches = [accs[i : i + batch_size] for i in range(0, len(accs), batch_size)]
    print("{}: {} proteins in {} batches of {}".format(
        label, len(accs), len(batches), batch_size))

    tic = time.time()
    records = []
    for i, batch in enumerate(fn.map(batches, kwargs={"force": force}), start=1):
        records.extend(batch)
        ok = sum(1 for r in records if r["status"] in ("ok", "cached"))
        print("  [{}/{}] {} done, {} ok ({:.1f} min)".format(
            i, len(batches), len(records), ok, (time.time() - tic) / 60.0))
        sys.stdout.flush()

    print("\n{} in {:.1f} min".format(label, (time.time() - tic) / 60.0))
    for status, n in Counter(r["status"] for r in records).most_common():
        print("  {:<20} {}".format(status, n))
    for record in [r for r in records if "log" in r][:3]:
        print("\n  {} ({}):".format(record["acc"], record["status"]))
        for line in record["log"].strip().splitlines()[-6:]:
            print("      " + line)
    return records


@app.local_entrypoint()
def prepare(limit: int = 0, batch_size: int = 8, force: bool = False):
    """Phase A: fetch and triangulate every protein (or the first --limit)."""
    sites = HERE / "lists/sites.csv"
    if not sites.exists():
        sys.exit("lists/sites.csv not found -- run prepare_dataset.py first")
    upload_lists.remote(sites.read_text(), (HERE / "lists/proteins.txt").read_text())
    run_phase(prepare_batch, accessions(limit), batch_size, force, "phase A (surfaces)")
    print("\nnext: modal run modal_run.py::patches")


@app.local_entrypoint()
def patches(limit: int = 0, batch_size: int = 8, force: bool = False):
    """Phase B: cut glycosite patches from the surfaces phase A built."""
    run_phase(patch_batch, accessions(limit), batch_size, force, "phase B (patches)")
    print("\nnext: modal run modal_run.py::descriptors")


@app.local_entrypoint()
def descriptors(force: bool = False):
    """Run the frozen network and build the feature table."""
    tic = time.time()
    out = descriptors_pass.remote(force=force)
    for key in ("descriptors", "table"):
        print("{} ok: {}".format(key, out[key + "_ok"]))
        for line in out[key + "_log"].strip().splitlines()[-20:]:
            print("  " + line)
    print("\ntotal {:.1f} min\nnext: modal run modal_run.py::fetch_features".format(
        (time.time() - tic) / 60.0))


@app.local_entrypoint()
def fetch_features():
    """Download the feature table so the models can be iterated on locally."""
    out = read_outputs.remote()
    if not out:
        sys.exit("nothing to download yet -- run ::descriptors first")
    for rel, blob in out.items():
        dest = HERE / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(blob)
        print("wrote {} ({:,} bytes)".format(dest, len(blob)))
    print("\nnow, locally:\n"
          "  python $MASIF/source/masif_glyco/train_mlp.py")


@app.local_entrypoint()
def status():
    info = volume_status.remote()
    total = len(accessions())
    print("volume masif-glyco:")
    print("  surfaces (.ply):    {:>6} / {}".format(info["surfaces"], total))
    print("  site patch dirs:    {:>6} / {}".format(info["site_patches"], total))
    print("  descriptors (.npz): {:>6} / {}".format(info["descriptors"], total))
    print("  feature table:      {}".format(info["has_feature_table"]))
