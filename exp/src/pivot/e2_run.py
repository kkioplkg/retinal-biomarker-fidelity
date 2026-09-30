"""E2 -- the main ReliSeg fine-tuning grid (pivot proposal v3, section E2).

Grid
----
    datasets  {hrf, fives}
    seeds     {0, 1, 2}   -- fine-tuned from ``runs/seg/<ds>/seed<k>/best.pt``
    configs   baseline       no fine-tune, re-inference of the seed checkpoint
                             through exactly the E2 path (parity control)
              cfloss_style   CF-Loss-style: density + FD on the mask with the
                             fixed 2..64 ladder, lambda = 1.0, no clDice
              reliseg_nocl   CE-Dice + density/length/adaptive-FD, lambda = 0.25
              reliseg        CE-Dice + 0.5*clDice + density/length/adaptive-FD,
                             lambda = 0.25

Ablation (HRF seed 0 only): ``reliseg`` minus each of {density, length, FD},
and ``reliseg`` with the fixed box ladder (adaptive-vs-fixed).

Layout
------
    runs/pivot/<ds>/seed<k>/<config>/{last.pt,best.pt,log.json,summary.json,
                                      pred/{prob,mask,manifest.csv,
                                            pixel_metrics.csv},
                                      bio.csv}

Execution
---------
Work is a single ordered job list; every worker claims jobs by atomically
creating a lock directory, so two GPU lanes (and a lane that joins late, when
GPU 0 frees up) share one queue with no coordination.  Jobs whose dependencies
are not satisfied yet are skipped and retried on the next pass.

    python -m src.pivot.e2_run plan
    python -m src.pivot.e2_run gpu  --gpu 1          # GPU lane worker
    python -m src.pivot.e2_run bio  --procs 12       # CPU biomarker stage
    python -m src.pivot.e2_run status
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Dict, List, Optional

EXP_ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", ".."))
PY = sys.executable
QUEUE_DIR = os.path.join("runs", "pivot", "_e2_queue")
LOG_DIR = os.path.join("runs", "pivot", "_e2_logs")

DATASETS = ("hrf", "fives")
SEEDS = (0, 1, 2)
EPOCHS = 50

#: ---- CMIG review point 16 (2026-09-18): extra seeds for the key null ----
#: "种子太少" -- the reviewer asks for 5-10 seeds behind ReliSeg vs the
#: same-budget ``continued`` control on FIVES.  Seeds 3-5 are added for those
#: two arms plus the ``baseline`` reference they are compared against (every
#: delta-r is a within-seed paired quantity, so a fine-tuned seed without its
#: own baseline row cannot enter the table).  ``cfloss`` and ``reliseg_nocl``
#: stay at seeds 0-2: they are secondary arms and the reviewer's point is
#: about the null, not about the whole grid.
#:
#: Nothing about the *estimator* changes -- same fidelity_and_delta, same
#: paired image bootstrap, same pooling rule (mean of the per-seed delta-r).
#: The seed set widens, which is exactly what point 16 asks for; the ragged
#: grid is handled by ``e2_analysis`` already (it pairs (ref, seed) only where
#: both exist) and is visible in ``coverage_table``.
EXTRA_SEEDS = (3, 4, 5)
EXTRA_SEED_UNITS = {"fives": ("baseline", "continued", "reliseg")}


def seeds_for(dataset: str, config: str) -> tuple:
    """Seeds of one (dataset, config) cell -- 0-2, plus 3-5 where registered."""
    if config in EXTRA_SEED_UNITS.get(dataset, ()):
        return tuple(SEEDS) + tuple(EXTRA_SEEDS)
    return tuple(SEEDS)
#: FIVES has 90 validation images; validating all of them after every epoch
#: costs more wall clock than the epoch itself.  Evaluation is on ``last.pt``
#: anyway (see the report), so the val set is only used to *record* the
#: best-val-Dice trajectory -- 12 evenly spaced images suffice for that.
VAL_LIMIT = {"hrf": 0, "fives": 12}

#: config -> fine-tuning arguments (None = no fine-tune, inference only)
#:
#: ``cfloss`` is the *published* CF-Loss objective, ported verbatim from
#: ``third_party/feature-loss`` and verified against it
#: (``python -m src.pivot.cf_loss verify``): beta*CE + alpha*L_FD + gamma*L_vd
#: with the published alpha/beta/gamma and the dyadic, image-size-scaled box
#: ladder with raw-count regression.  The earlier hand-written "fixed 2..64
#: ladder, log-log slope" variant is retired from the main grid and survives
#: only as the ``abl_fixed_ladder`` row.
CONFIGS: Dict[str, Optional[dict]] = {
    "baseline": None,
    # same-budget control (DECISIONS 2026-09-17 13:30): identical starting
    # checkpoint, epochs, LR schedule and batch size as every other arm, but
    # the base loss only -- lam = 0 zeroes the measurement block, so the
    # surrogates are still computed and logged and contribute no gradient.
    # Delta-r is reported against BOTH this arm and the untouched baseline.
    "continued": dict(loss_kind="measure", lam=0.0, w_cldice=0.5,
                      terms="density,length,fd", fd_ladder="adaptive"),
    "cfloss": dict(loss_kind="cf_faithful", lam=1.0, w_cldice=0.0,
                   terms="density,fd", fd_ladder="fixed", cf_scale=1.0),
    "reliseg_nocl": dict(loss_kind="measure", lam=0.25, w_cldice=0.0,
                         terms="density,length,fd", fd_ladder="adaptive"),
    "reliseg": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                    terms="density,length,fd", fd_ladder="adaptive"),
}

#: every config evaluated on both datasets, in report order
MAIN_CONFIGS = ("baseline", "continued", "reliseg", "reliseg_nocl",
                "cfloss")

#: HRF seed-0 ablation of the default config (ReliSeg, clDice kept)
ABLATIONS: Dict[str, dict] = {
    "abl_no_density": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                           terms="length,fd", fd_ladder="adaptive"),
    "abl_no_length": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                          terms="density,fd", fd_ladder="adaptive"),
    "abl_no_fd": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                      terms="density,length", fd_ladder="adaptive"),
    # single-term arms (DECISIONS 2026-09-17 13:30): the leave-one-out rows
    # above cannot say which term *carries* the effect, only which is
    # dispensable.  clDice kept, lam and ladder unchanged.
    "abl_density_only": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                             terms="density", fd_ladder="adaptive"),
    "abl_length_only": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                            terms="length", fd_ladder="adaptive"),
    "abl_fd_only": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                        terms="fd", fd_ladder="adaptive"),
    # adaptive vs fixed box ladder (the retired CF-Loss-style 2..64 ladder)
    "abl_fixed_ladder": dict(loss_kind="measure", lam=0.25, w_cldice=0.5,
                             terms="density,length,fd", fd_ladder="fixed"),
    # the published feature terms on *our* base loss: isolates the terms from
    # the base objective (published CF-Loss drops Dice and clDice entirely)
    "abl_cf_on_base": dict(loss_kind="cf_on_base", lam=1.0, w_cldice=0.5,
                           terms="density,fd", fd_ladder="fixed", cf_scale=1.0),
    # ... and the same with the block rescaled to ~25 % of the loss at
    # initialisation, the way lambda was calibrated for ReliSeg: the control
    # for "the baseline only lost because it was under-weighted"
    "abl_cf_matched": dict(loss_kind="cf_on_base", lam=1.0, w_cldice=0.5,
                           terms="density,fd", fd_ladder="fixed", cf_scale=11.0),
}

#: zero-shot cross-dataset fidelity: every main-grid checkpoint is applied,
#: untouched, to two datasets neither model ever saw.  Inference runs at the
#: *source* model's resolution convention (longest side 1536, patch 768).
EXTERNAL = ("chasedb1", "stare")
SRC_LONGEST, SRC_PATCH = 1536, 768

#: checkpoints evaluated per fine-tuned run
EVAL_CKPTS = ("last", "fid")

#: ---- E3: external (mask-less) datasets, DECISIONS 2026-09-17 02:00 ----
#: order is frozen: IDRiD -> Messidor-2 -> APTOS-2019 -> ODIR-512 (sensitivity)
E3_DATASETS = ("idrid", "messidor2", "aptos2019", "odir5k")
#: inference scale for the external sets: the training convention (longest 1536)
E3_LONGEST = 1536


def e3_checkpoints() -> List[tuple]:
    """(dataset, seed, config, ckpt_path) for every frozen E3 segmenter.

    Frozen scope (DECISIONS 2026-09-17 02:00): FIVES seeds 0-2 x
    {baseline, ReliSeg, faithful CF-Loss} (primary) then HRF seed 0 x
    {baseline, ReliSeg} (development-set models, secondary).  This is the
    *protocol* scope; who runs which part is decided by :func:`e3_jobs`.
    """
    out: List[tuple] = []
    for k in SEEDS:
        for cfg in ("baseline", "reliseg", "cfloss"):
            out.append(("fives", k, cfg, eval_ckpt("fives", k, cfg, "last")))
    for cfg in ("baseline", "reliseg"):
        out.append(("hrf", 0, cfg, eval_ckpt("hrf", 0, cfg, "last")))
    return out


#: E3 work owned by the LAN node (LAN_HOST, one 3080), set up by a
#: separate job on 2026-09-17 12:05: every FIVES-trained checkpoint on the
#: three primary/replication cohorts.  Its results rsync back into
#: ``results/pivot/e3/<ext>/<tag>/``.  These pairs are NOT queued locally --
#: two nodes writing the same resumable ``bio.csv`` would interleave rows.
E3_REMOTE_COHORTS = ("idrid", "messidor2", "aptos2019")
E3_REMOTE_SOURCE = "fives"


def e3_is_remote(ext: str, ds: str) -> bool:
    return ext in E3_REMOTE_COHORTS and ds == E3_REMOTE_SOURCE


def e3_local_units() -> List[tuple]:
    """(external set, dataset, seed, config, ckpt) this node runs, in order.

    Local GPUs are reserved (team decision, 2026-09-17 12:05) for the E2 work
    first; what is left of E3 for this node is
      1. HRF seed 0 x {baseline, ReliSeg} on the three primary cohorts -- the
         LAN node only carries the FIVES-trained checkpoints; and
      2. the whole ODIR-512 sensitivity cohort, which the LAN node does not
         carry at all.
    """
    out: List[tuple] = []
    for ext in E3_DATASETS:
        for ds, k, cfg, ck in e3_checkpoints():
            if e3_is_remote(ext, ds):
                continue
            out.append((ext, ds, k, cfg, ck))
    return out


# --------------------------------------------------------------------------
def run_dir(ds: str, seed: int, config: str) -> str:
    return os.path.join("runs", "pivot", ds, f"seed{seed}", config)


def tag_of(ds: str, seed: int, config: str) -> str:
    return f"e2_{ds}_s{seed}_{config}"


def init_ckpt(ds: str, seed: int) -> str:
    return os.path.join("runs", "seg", ds, f"seed{seed}", "best.pt")


def eval_ckpt(ds: str, seed: int, config: str, which: str = "last") -> str:
    """Checkpoint E2 evaluates.

    ``last`` -- the model after the full 50-epoch schedule (the pre-specified
    primary; val Dice does not track the objective, see the report).
    ``fid``  -- the epoch with the best *validation measurement fidelity*
    (mean r over density/length/FD across the val images, or the mean relative
    surrogate error when the val split is too small for a correlation).  Never
    selected on anything from a test split.
    """
    if config == "baseline":
        return init_ckpt(ds, seed)
    return os.path.join(run_dir(ds, seed, config),
                        "last.pt" if which == "last" else "best_fid.pt")


def _train_argv(ds: str, seed: int, config: str, spec: dict) -> List[str]:
    return ["-m", "src.pivot.p5_finetune",
            "--dataset", ds, "--seed", str(seed),
            "--ckpt", init_ckpt(ds, seed),
            "--out", run_dir(ds, seed, config),
            "--loss-kind", spec["loss_kind"], "--lam", str(spec["lam"]),
            "--w-cldice", str(spec["w_cldice"]),
            "--terms", spec["terms"], "--fd-ladder", spec["fd_ladder"],
            "--cf-scale", str(spec.get("cf_scale", 1.0)),
            "--epochs", str(EPOCHS),
            "--val-limit", str(VAL_LIMIT[ds])]


def _infer_job(ds: str, k: str, cfg: str, which: str, target: str = None) -> dict:
    """One inference job.  ``target`` = None means the model's own test split;
    otherwise the zero-shot external dataset, run at the source scale."""
    d = run_dir(ds, k, cfg)
    ck = eval_ckpt(ds, k, cfg, which)
    suff = "" if which == "last" else "_fid"
    if target is None:
        root = os.path.join(d, "pred" + suff)
        tag = tag_of(ds, k, cfg) + suff
        argv = ["-m", "src.pivot.p5_eval", "infer", "--dataset", ds,
                "--ckpt", ck, "--tag", tag, "--root", root]
        jid = f"infer:{ds}:s{k}:{cfg}{suff}"
    else:
        root = os.path.join(d, f"pred{suff}_{target}")
        tag = tag_of(ds, k, cfg) + suff + "_" + target
        argv = ["-m", "src.pivot.p5_eval", "infer", "--dataset", target,
                "--ckpt", ck, "--tag", tag, "--root", root,
                "--infer-longest", str(SRC_LONGEST),
                "--infer-patch", str(SRC_PATCH)]
        jid = f"infer:{ds}:s{k}:{cfg}{suff}->{target}"
    return dict(id=jid, kind="infer", dataset=ds, seed=k, config=cfg,
                which=which, target=target, tag=tag, root=root,
                gpu_needed=True, argv=argv, deps=[ck],
                done=os.path.join(root, "infer_meta.json"))


def eval_units() -> List[tuple]:
    """(dataset, seed, config, which) for every checkpoint E2 evaluates."""
    out = []
    for ds in ("fives", "hrf"):
        for cfg in MAIN_CONFIGS:
            for k in seeds_for(ds, cfg):
                out.append((ds, k, cfg, "last"))
                if cfg != "baseline":
                    out.append((ds, k, cfg, "fid"))
    for cfg in ABLATIONS:
        out.append(("hrf", 0, cfg, "last"))
        out.append(("hrf", 0, cfg, "fid"))
    return out


def build_jobs() -> List[dict]:
    """Ordered job list, longest first.  Each job: id, kind, argv, deps, done."""
    jobs: List[dict] = []

    # the parity baselines need no training and are cheap; running them first
    # lets the CPU biomarker stage start while the GPU lanes are still training
    for ds in ("fives", "hrf"):
        for k in seeds_for(ds, "baseline"):
            jobs.append(_infer_job(ds, k, "baseline", "last"))

    # ---- training, longest first ----
    combos: List[tuple] = []
    for ds in ("fives", "hrf"):
        for cfg in ("continued", "reliseg", "reliseg_nocl", "cfloss"):
            for k in seeds_for(ds, cfg):
                combos.append((ds, k, cfg, CONFIGS[cfg]))
    for cfg, spec in ABLATIONS.items():
        combos.append(("hrf", 0, cfg, spec))
    for ds, k, cfg, spec in combos:
        d = run_dir(ds, k, cfg)
        jobs.append(dict(id=f"train:{ds}:s{k}:{cfg}", kind="train",
                         dataset=ds, seed=k, config=cfg, gpu_needed=True,
                         argv=_train_argv(ds, k, cfg, spec),
                         deps=[init_ckpt(ds, k)],
                         done=os.path.join(d, "summary.json")))

    # ---- in-domain inference of every evaluated checkpoint ----
    for ds, k, cfg, which in eval_units():
        if cfg == "baseline":
            continue                      # already queued at the top
        jobs.append(_infer_job(ds, k, cfg, which))

    # ---- zero-shot cross-dataset arm (main grid, last.pt only) ----
    for target in EXTERNAL:
        for ds in ("fives", "hrf"):
            for cfg in MAIN_CONFIGS:
                for k in SEEDS:
                    jobs.append(_infer_job(ds, k, cfg, "last", target=target))

    # ---- E3: external mask-less datasets (LOWEST priority: appended last) ----
    jobs += e3_jobs()
    return jobs


def e3_jobs() -> List[dict]:
    """``src.pivot.e3_external bio`` for the external work THIS node owns.

    The FIVES-trained checkpoints on IDRiD / Messidor-2 / APTOS-2019 run on
    the LAN node instead (see :data:`E3_REMOTE_COHORTS`) and are deliberately
    absent here.

    Lowest priority by construction -- the worker walks the list in order, so
    these only start once every E2 train/infer job is done or claimed.

    Done-marker is ``meta.json``, *not* ``bio.csv``: the stage is resumable and
    appends to ``bio.csv`` every 10 images, so ``bio.csv`` exists long before
    the pass is complete, whereas ``meta.json`` is written only after the last
    image.  A killed job therefore resumes instead of being mistaken for done.
    """
    out: List[dict] = []
    for ext, ds, k, cfg, ck in e3_local_units():
            tag = f"e3_{ds}_s{k}_{cfg}"
            d = os.path.join("results", "pivot", "e3", ext, tag)
            out.append(dict(
                id=f"e3:{ext}:{ds}:s{k}:{cfg}", kind="e3bio",
                dataset=ds, seed=k, config=cfg, which="last", target=ext,
                tag=tag, root=d, gpu_needed=True, gpu_flag="--device",
                gpu_value="cuda:{gpu}",
                argv=["-m", "src.pivot.e3_external", "bio",
                      "--dataset", ext, "--ckpt", ck, "--tag", tag,
                      "--resize-longest", str(E3_LONGEST),
                      "--resolution-record"],
                deps=[ck],
                done=os.path.join(d, "meta.json")))
    return out


def bio_jobs() -> List[dict]:
    jobs = []
    # GT biomarkers for the external sets (bio_master has no STARE rows)
    for target in EXTERNAL:
        jobs.append(dict(id=f"gtbio:{target}", kind="gtbio", gpu_needed=False,
                         argv=["-m", "src.pivot.p5_eval", "gtbio",
                               "--dataset", target, "--tag", "gt_e2"],
                         deps=[],
                         done=os.path.join("results", "pivot", f"e2_gt_{target}.csv")))
    for j in build_jobs():
        if j["kind"] != "infer":
            continue
        jobs.append(dict(id="bio" + j["id"][5:], kind="bio",
                         dataset=j["dataset"], seed=j["seed"], config=j["config"],
                         which=j["which"], target=j["target"], gpu_needed=False,
                         argv=["-m", "src.pivot.p5_eval", "bio",
                               "--dataset", j["target"] or j["dataset"],
                               "--tag", j["tag"], "--root", j["root"]],
                         deps=[j["done"]],
                         done=os.path.join(j["root"], "bio.csv")))
    return jobs


# --------------------------------------------------------------------------
def _safe(job_id: str) -> str:
    """Job id -> NTFS-safe file stem ('>' is illegal in Windows file names)."""
    return job_id.replace(":", "__").replace(">", "_to_")


def claim(job_id: str) -> bool:
    """Atomic claim via ``os.mkdir`` (works on NTFS and across processes)."""
    os.makedirs(QUEUE_DIR, exist_ok=True)
    lock = os.path.join(QUEUE_DIR, _safe(job_id) + ".lock")
    try:
        os.mkdir(lock)
    except FileExistsError:
        return False
    with open(os.path.join(lock, "owner.json"), "w", encoding="utf-8") as f:
        json.dump({"pid": os.getpid(), "t": time.strftime("%Y-%m-%d %H:%M:%S")}, f)
    return True


def release(job_id: str) -> None:
    lock = os.path.join(QUEUE_DIR, _safe(job_id) + ".lock")
    try:
        os.remove(os.path.join(lock, "owner.json"))
        os.rmdir(lock)
    except OSError:
        pass


def deps_ok(job: dict) -> bool:
    return all(os.path.exists(p) for p in job["deps"])


def is_done(job: dict) -> bool:
    return os.path.exists(job["done"])


def run_job(job: dict, gpu: Optional[int], extra: List[str]) -> int:
    os.makedirs(LOG_DIR, exist_ok=True)
    log = os.path.join(LOG_DIR, _safe(job["id"]) + ".log")
    argv = list(job["argv"]) + list(extra)
    if gpu is not None and job["gpu_needed"]:
        # most stages take ``--gpu <int>``; e3_external takes ``--device cuda:<int>``
        flag = job.get("gpu_flag", "--gpu")
        val = job.get("gpu_value", "{gpu}").format(gpu=gpu)
        argv += [flag, val]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    print(f"[e2] >>> {job['id']}\n     {' '.join(argv)}\n     log {log}", flush=True)
    t0 = time.time()
    with open(log, "a", encoding="utf-8") as fh:
        fh.write(f"\n===== {time.strftime('%Y-%m-%d %H:%M:%S')} "
                 f"{' '.join(argv)} =====\n")
        fh.flush()
        rc = subprocess.call([PY] + argv, cwd=EXP_ROOT, env=env,
                             stdout=fh, stderr=subprocess.STDOUT)
    dt = time.time() - t0
    print(f"[e2] <<< {job['id']} rc={rc} in {dt/60:.1f} min", flush=True)
    return rc


def worker(jobs: List[dict], gpu: Optional[int], extra: List[str]) -> int:
    fails = []
    while True:
        did_work = False
        pending = False
        for job in jobs:
            if is_done(job):
                continue
            if job["id"] in fails:
                continue
            if not deps_ok(job):
                pending = True
                continue
            if not claim(job["id"]):
                pending = True
                continue
            rc = run_job(job, gpu, extra)
            if rc != 0 or not is_done(job):
                print(f"[e2] FAILED {job['id']} rc={rc}", flush=True)
                fails.append(job["id"])
                release(job["id"])
            did_work = True
        if not did_work:
            if not pending:
                break
            print("[e2] nothing runnable; waiting 120 s for the other lane",
                  flush=True)
            time.sleep(120)
            if all(is_done(j) or j["id"] in fails for j in jobs):
                break
    left = [j["id"] for j in jobs if not is_done(j)]
    print(f"[e2] worker done. failures={fails} not-done={left}", flush=True)
    return 1 if fails else 0


# --------------------------------------------------------------------------
def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "status"):
        s = sub.add_parser(name)
        s.add_argument("--all", action="store_true")
    g = sub.add_parser("gpu")
    g.add_argument("--gpu", type=int, required=True)
    g.add_argument("--only", default="", help="substring filter on job ids")
    b = sub.add_parser("bio")
    b.add_argument("--procs", type=int, default=12)
    a = ap.parse_args(argv)

    os.chdir(EXP_ROOT)
    jobs = build_jobs()
    bios = bio_jobs()

    if a.cmd in ("plan", "status"):
        for j in jobs + bios:
            state = "DONE " if is_done(j) else ("READY" if deps_ok(j) else "WAIT ")
            print(f"{state} {j['id']}")
        n = sum(is_done(j) for j in jobs + bios)
        print(f"\n{n}/{len(jobs) + len(bios)} complete "
              f"({sum(1 for j in jobs if j['kind']=='train')} train, "
              f"{sum(1 for j in jobs if j['kind']=='infer')} infer, "
              f"{len(bios)} bio)")
        return 0

    if a.cmd == "gpu":
        sel = [j for j in jobs if (not a.only or a.only in j["id"])]
        return worker(sel, a.gpu, [])
    if a.cmd == "bio":
        return worker(bios, None, ["--procs", str(a.procs)])
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
