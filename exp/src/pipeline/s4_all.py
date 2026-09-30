"""One orchestrator for the whole of stage S4 (+ S5 / S6 / S7 tail).

    cd exp
    python -m src.pipeline.s4_all --dry-run          # print every command
    python -m src.pipeline.s4_all --status           # what is done / pending
    python -m src.pipeline.s4_all                    # run everything
    python -m src.pipeline.s4_all --only D,E --datasets drive,hrf --seeds 0

The unit of work is a :class:`Step`: a name, a command, the files it must
produce, the lane it runs in, and the steps it depends on.  Everything the
orchestrator does follows from those four fields:

*resumability*  a step whose declared outputs all exist is skipped (``--force``
                re-runs it anyway).  Nothing is inferred from timestamps -- if
                you change a config value that should invalidate a step, delete
                its output or pass ``--force``.
*scheduling*    three lanes: ``gpu0`` and ``gpu1``, one process each, and a
                ``cpu`` pool of ``--jobs`` processes.  A GPU lane pins its
                children with ``CUDA_VISIBLE_DEVICES``; CPU-lane children get
                ``CUDA_VISIBLE_DEVICES=""`` so a stray ``--gpu 0`` cannot creep
                onto a busy card.  Datasets are assigned to lanes exactly as the
                plan's S2/S3 grid did: GPU0 = DRIVE + HRF, GPU1 = CHASE + FIVES.
*ordering*      a topological order over the declared dependencies; a step runs
                only once every dependency has finished successfully.  A failed
                step marks its dependants ``BLOCKED`` instead of running them on
                missing inputs.

The dependency graph (letters are the plan's S4 items; ``->`` reads "feeds")::

    A head            -> B data -> C models -> D main runs -----------\\
                                            \\-> F ablations ----------|
                                            \\-> H exp2 ---------------|
    (seg_oof only)    -> E baselines -----------------------------> J make_all
    G lodo head -> G lodo data -> G lodo models -> G lodo runs -----/
    (seg ckpts)       -> I robustness -----------------------------/

Anti-leakage rules enforced by construction (not by convention):

* every training artefact is built from the **training split** only, and from
  **out-of-fold** predictions of those images (``runs/seg_oof/<ds>/pred``), so
  no model was ever fitted on the image whose prediction it consumes;
* test predictions (``runs/seg/<ds>/seed<k>/pred``) are read at evaluation time
  only, and are always **seed-matched** to the head / scorer used with them;
* the LODO block lists only the three *source* datasets everywhere -- head,
  scorer, deployed ``R_false`` and (when ``runs/btr/btr_*_wo_<ds>.joblib``
  exists) the BTR heads themselves.

Timings are documented in ``exp/S4_RUNBOOK.md``.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
EXP_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))

LETTERS = "ABCDEFGHIJ"

#: ``src.c1.btr`` names its leave-one-dataset-out heads after the **C1 event
#: table's** ``dataset`` column, which carries the upper-case dataset names, so
#: ``runs/btr/.../btr_miss_wo_CHASE_DB1.joblib`` -- not the lower-case keys the
#: rest of S4 uses.  Getting this wrong makes ``run_rigr`` fail to find the head
#: and silently fall back to the all-data one.
C1_DATASET_NAMES = {
    "drive": "DRIVE", "chasedb1": "CHASE_DB1", "hrf": "HRF",
    "fives": "FIVES", "stare": "STARE",
}

#: Tab.3 ablation ladder.  Each entry is (variant name, run_rigr overrides,
#: which scorer/data variant it draws on).  ``omega0`` picks the data built
#: with *uniform-only* severances, which is what "+failcond (omega=0.5 vs 0)"
#: contrasts against.
ABLATIONS = [
    # name,              mode,      evidence, orientation, mu,     data
    ("appearance_only",  "uniform", "frangi", "uniform",   0.0,    "omega0"),
    ("+orientation",     "uniform", "head",   "learned",   0.0,    "omega0"),
    ("+curvature",       "uniform", "head",   "learned",   None,   "omega0"),
    ("+failcond",        "uniform", "head",   "learned",   None,   "main"),
    ("+btr_utility",     "risk",    "head",   "learned",   None,   "main"),
    ("frangi_vs_learned", "risk",   "frangi", "uniform",   None,   "main"),
]

DEFAULTS: Dict[str, Any] = dict(
    python=sys.executable,
    datasets=["drive", "chasedb1", "hrf", "fives"],
    seeds=[0, 1, 2],
    modes=["prob", "uniform", "risk"],
    gpu_lanes={"0": ["drive", "hrf"], "1": ["chasedb1", "fives"]},
    cpu_jobs=4,
    # roots
    seg_root=os.path.join("runs", "seg"),
    oof_root=os.path.join("runs", "seg_oof"),
    head_root=os.path.join("runs", "rigr_head"),
    data_root=os.path.join("runs", "rigr_data"),
    model_root=os.path.join("runs", "rigr_models"),
    rigr_root=os.path.join("runs", "rigr"),
    repair_root=os.path.join("runs", "repair"),
    btr_root=os.path.join("runs", "btr"),
    results_dir="results",
    log_dir=os.path.join("runs", "s4_logs"),
    # A
    head_epochs=60,
    head_iters=100,
    head_workers=4,          # DataLoader workers per head training
    head_threads=4,          # torch intra-op threads in the training process
    # B
    omega=0.5,
    n_cuts=15,
    # Deployed R_false biomarker pipelines.  "both" matches the
    # habs_trainonly R_miss head, which medians over pvbm+skan for FD /
    # density / total_length; a skan-only R_false would put the two terms of
    # U(e) = p R_miss - lambda (1-p) R_false on subtly different scales.
    # Costs roughly 2x in step B (PVBM's multifractal FD is the slow part).
    rfalse_bio="both",
    # per-image pool for step B; the cost is the two biomarker pipelines, not
    # the head, so the workers are CPU-only and single-threaded
    rfalse_workers=6,
    # PVBM multifractal rotations: 5 is the C1 setting (DECISIONS.md -- bit
    # identical to 25 and 2-3x faster)
    rfalse_fd_rotations=5,
    # how many images per dataset contribute R_false rows (the scorer pairs
    # still come from every image).  40 x 24 candidates is already ~1000
    # measured edits for FIVES, far more than the head can use, and it is what
    # takes step B on FIVES from ~22 h to well under an hour.
    rfalse_max_images={"fives": 40},
    max_cand_bio=24,
    # C
    val_fraction=0.3,
    scorer_model="logreg",
    # D / F -- the deployed BTR heads.
    # DECISIONS.md (2026-09-03 08:30) fixes them as: target H_abs, fitted on the
    # TRAINING split only, in-domain.  runs/btr/habs_trainonly/btr_miss_all.joblib
    # is the R_miss head; the R_false head is the per-(dataset, seed) deployed one
    # from step C, which src.rigr.fit_models now also fits on H_abs so the two can
    # be paired (BTRBackend refuses to mix target families).
    # The leak this avoids: runs/btr/btr_*_all.joblib (the flat, H_dep, all-events
    # layout) was fitted on perturbations of the *test* images RiGR is applied to.
    btr_runs_dir=os.path.join("runs", "btr", "habs_trainonly"),
    btr_variant="all",
    btr_target="habs",
    # D
    lam=1.0, eta=0.1, tau=0.5,
    bio="both",
    sweep_bio="skan",
    # Which seeds get --sweep in step D.  None = every seed (the original
    # behaviour).  [0] = Fig.3 curves and the matched-FCR statistics come
    # from seed 0 only; seeds 1 and 2 run at the default operating point
    # (lambda=1, tau=0.5), which is what Tab.2's main numbers use anyway.
    # Coordinator ruling 2026-09-05 22:30, see DECISIONS.md.
    sweep_seeds=None,
    sweep_lam="0.5,1,2,4",
    # Coordinator ruling 2026-09-05 23:20: 4 tau values -> a 16-cell grid
    # for prob (the only mode that sweeps tau), down from 28.
    sweep_tau="0.3,0.5,0.7,0.9",
    # Datasets whose SWEEP-CELL biomarkers are measured at the C1 working
    # resolution (longest side 1536) with the 1/s native-unit conversion,
    # exactly as C1 and build_data do.  The main operating point in
    # per_image.csv stays at native resolution for every dataset.
    sweep_scale_work_datasets=["hrf", "fives"],
    # E
    rnca_iters=20000,
    evapore_epochs=150,
    baseline_train_seed=0,
    baseline_methods=["geometric", "rnca", "evapore"],
    baseline_infer_gpu=False,
    # G
    lodo_head_epochs=60,
    lodo_modes=["risk", "uniform"],
    # Which BTR heads price the LODO runs.  DECISIONS.md (2026-09-03 08:30)
    # fixes the deployed heads as H_abs fitted on the TRAINING split only, i.e.
    # runs/btr/habs_trainonly/btr_{miss,false}_wo_<D>.joblib -- RiGR is applied
    # to predicted masks of the *test* images, so a head fitted on
    # perturbations of those same images has already seen the answer.
    lodo_btr_runs_dir=os.path.join("runs", "btr", "habs_trainonly"),
    # DECISIONS.md 2026-09-15 21:30 item (1) / 2026-09-16 00:05: the LODO
    # R_false is now the DEPLOYED definition -- counterfactual harm on the A*
    # candidate path -- so that it matches the main text, and the C1 bridge head
    # btr_false_wo_<D>.joblib is kept for the Exp1 analysis only.  The head is
    # runs/rigr_models/lodo_<D>/seed0/btr_false_deployed.joblib, which G:fit
    # already writes from runs/rigr_data/lodo_<D>/rfalse_train.csv, i.e. the
    # three SOURCE domains scored with the LODO head (held-out domain absent).
    # The old comment here claimed this head targets Hdep_* and so could not be
    # paired with the habs_trainonly R_miss: that is stale.  ``fit_models``
    # defaults to --rfalse_target habs, and all four heads carry
    # meta['target'] == 'habs' (drive 712 rows / chasedb1 593 / hrf 551 /
    # fives 245); BTRBackend accepts the pair -- verified by the 2026-09-15
    # re-runs of G:run:<D>:risk on the LAN node.
    lodo_btr_deployed_false=True,
    # H
    exp2_bio="skan",
    exp2_max_cand=40,
    exp2_workers=max(1, (os.cpu_count() or 4) // 2),
    # I
    stability_K=8,
    robust_seed=0,
    resolution_dataset="hrf",
    # which letters run on a GPU lane (everything else goes to the CPU pool)
    gpu_steps=["A", "B", "E_train", "G_head", "G_data", "I"],
    # ---- image caps -------------------------------------------------------
    # FIVES has 510 training and 200 test images against DRIVE's 17/20, so it
    # would otherwise account for ~90% of the S4 compute and dominate any
    # dataset-averaged table.  The plan already subsets FIVES for C1 ("FIVES
    # 60/100 image subset", exp/DECISIONS.md); these caps apply the same policy
    # to S4.  The cap takes the FIRST n images of the fixed split, so **every**
    # method -- RiGR, the three baselines, the ablations, Exp2 and the
    # robustness runs -- sees exactly the same image set.
    # Set a value to null to use the whole split.
    image_caps={"train": {"fives": 120}, "test": {"fives": 60}},
)


# --------------------------------------------------------------------------
@dataclass
class Step:
    key: str
    letter: str
    cmd: List[str]
    outputs: List[str]
    lane: str = "cpu"                       # 'cpu' | 'gpu0' | 'gpu1'
    deps: List[str] = field(default_factory=list)
    env: Dict[str, str] = field(default_factory=dict)
    note: str = ""
    #: matched by --skip-steps: this invocation never launches it.  Its
    #: outputs are still checked, so the moment they appear (a checkpoint
    #: copied in from a remote run, say) it counts as DONE and its dependants
    #: become runnable -- in this invocation if the file is already there, in
    #: a later one otherwise.  A dependant of a skipped step is left PENDING,
    #: never BLOCKED: nothing has failed, the input simply has not arrived.
    skip: bool = False

    def done(self) -> bool:
        return bool(self.outputs) and all(
            os.path.exists(os.path.join(EXP_ROOT, o)) for o in self.outputs)

    def missing(self) -> List[str]:
        return [o for o in self.outputs
                if not os.path.exists(os.path.join(EXP_ROOT, o))]


def _q(a: str) -> str:
    return '"%s"' % a if (" " in a or '"' in a) else a


def shell(step: Step) -> str:
    pre = " ".join("%s=%s" % (k, v or "''") for k, v in sorted(step.env.items()))
    return (pre + " " if pre else "") + " ".join(_q(a) for a in step.cmd)


# --------------------------------------------------------------------------
class Planner:
    """Builds the S4 step graph from a config dict."""

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.py = cfg["python"]
        self.steps: "Dict[str, Step]" = {}

    # -- helpers ---------------------------------------------------------
    def lane_of(self, ds: str) -> str:
        for gpu, names in self.cfg["gpu_lanes"].items():
            if ds in names:
                return "gpu%s" % gpu
        return "gpu0"

    def _lane(self, letter_key: str, ds: str) -> str:
        return self.lane_of(ds) if letter_key in self.cfg["gpu_steps"] else "cpu"

    def cap(self, kind: str, ds: str) -> Optional[int]:
        """Image cap for ``kind`` in ('train', 'test') on dataset ``ds``."""
        v = (self.cfg.get("image_caps") or {}).get(kind, {}).get(ds)
        return int(v) if v else None

    def cap_args(self, kind: str, ds: str, flag: str = "--limit") -> List[str]:
        n = self.cap(kind, ds)
        return [flag, str(n)] if n else []

    def add(self, step: Step) -> Step:
        if step.key in self.steps:
            raise ValueError("duplicate step key %r" % step.key)
        self.steps[step.key] = step
        return step

    def _mod(self, module: str, *args: str) -> List[str]:
        return [self.py, "-m", module] + [str(a) for a in args]

    # -- paths -----------------------------------------------------------
    def head_dir(self, ds: str, seed: int) -> str:
        return os.path.join(self.cfg["head_root"], ds, "seed%d" % seed)

    def head_ckpt(self, ds: str, seed: int) -> str:
        return os.path.join(self.head_dir(ds, seed), "best.pt")

    def data_dir(self, ds: str, variant: str = "main") -> str:
        suffix = "" if variant == "main" else "_w0"
        return os.path.join(self.cfg["data_root"], ds + suffix)

    def model_dir(self, ds: str, seed: int, variant: str = "main") -> str:
        suffix = "" if variant == "main" else "_w0"
        return os.path.join(self.cfg["model_root"], ds + suffix, "seed%d" % seed)

    def pred_dir(self, ds: str, seed: int) -> str:
        return os.path.join(self.cfg["seg_root"], ds, "seed%d" % seed, "pred")

    def oof_dir(self, ds: str) -> str:
        return os.path.join(self.cfg["oof_root"], ds, "pred")

    def seg_ckpt(self, ds: str, seed: int) -> str:
        return os.path.join(self.cfg["seg_root"], ds, "seed%d" % seed, "best.pt")

    # ------------------------------------------------------------------ A
    def plan_A(self, datasets, seeds):
        c = self.cfg
        for ds in datasets:
            for k in seeds:
                out = self.head_dir(ds, k)
                self.add(Step(
                    key="A:%s:s%d" % (ds, k), letter="A",
                    cmd=self._mod("src.rigr.train_head",
                                  "--dataset", ds, "--seed", k, "--gpu", 0,
                                  "--epochs", c["head_epochs"],
                                  "--iters", c["head_iters"],
                                  "--workers", c["head_workers"],
                                  "--threads", c["head_threads"],
                                  "--pred_dir", self.oof_dir(ds),
                                  "--out", out,
                                  # same image set (and the same per-image
                                  # target cache) as step B
                                  *self.cap_args("train", ds)),
                    outputs=[os.path.join(out, "best.pt"),
                             os.path.join(out, "history.json")],
                    lane=self._lane("A", ds),
                    note="micro head on [I, P_oof]; P is the OUT-OF-FOLD "
                         "prediction of the training images"))

    # ------------------------------------------------------------------ B
    def plan_B(self, datasets, seeds, want_omega0: bool):
        c = self.cfg
        for ds in datasets:
            prev = None
            for k in seeds:
                out = self.data_dir(ds)
                rfalse = (k == min(seeds))
                args = ["--dataset", ds, "--seed", k, "--out", out,
                        "--oof_dir", self.oof_dir(ds),
                        "--head_ckpt", self.head_ckpt(ds, k), "--gpu", 0,
                        "--omega", c["omega"], "--n_cuts", c["n_cuts"],
                        "--bio", c["rfalse_bio"],
                        "--max_cand_bio", c["max_cand_bio"],
                        "--workers", c["rfalse_workers"],
                        "--fd_rotations", c["rfalse_fd_rotations"]]
                args += self.cap_args("train", ds)
                if rfalse and c.get("rfalse_max_images", {}).get(ds):
                    args += ["--rfalse_max_images",
                             "%s=%d" % (ds, int(c["rfalse_max_images"][ds]))]
                if rfalse:
                    args.append("--rfalse")
                outs = [os.path.join(out, "seed%d" % k, "pairs.npz"),
                        os.path.join(out, "failure_prior.json")]
                if rfalse:
                    outs.append(os.path.join(out, "rfalse_train.csv"))
                deps = ["A:%s:s%d" % (ds, k)]
                if prev:                       # pi is fitted once, by the first
                    deps.append(prev)          # seed; the others reuse the file
                st = self.add(Step(
                    key="B:%s:s%d" % (ds, k), letter="B",
                    cmd=self._mod("src.rigr.build_data", *args),
                    outputs=outs, deps=deps, lane=self._lane("B", ds),
                    note="pi from OOF preds + scorer pairs"
                         + (" + deployed-R_false table" if rfalse else "")))
                prev = st.key

            if want_omega0:
                k0 = min(seeds)
                out0 = self.data_dir(ds, "omega0")
                self.add(Step(
                    key="B:%s:s%d:w0" % (ds, k0), letter="B",
                    cmd=self._mod("src.rigr.build_data",
                                  "--dataset", ds, "--seed", k0, "--out", out0,
                                  "--oof_dir", self.oof_dir(ds),
                                  "--head_ckpt", self.head_ckpt(ds, k0),
                                  "--gpu", 0, "--omega", 0.0,
                                  "--n_cuts", c["n_cuts"],
                                  "--workers", c["rfalse_workers"],
                                  *self.cap_args("train", ds)),
                    outputs=[os.path.join(out0, "seed%d" % k0, "pairs.npz")],
                    deps=["A:%s:s%d" % (ds, k0)], lane=self._lane("B", ds),
                    note="omega = 0 (uniform severances only): the control arm "
                         "of the Tab.3 '+failcond' contrast"))

    # ------------------------------------------------------------------ C
    def plan_C(self, datasets, seeds, want_omega0: bool):
        c = self.cfg
        for ds in datasets:
            k0 = min(seeds)
            for k in seeds:
                out = self.model_dir(ds, k)
                deps = ["B:%s:s%d" % (ds, k)]
                if k != k0:
                    deps.append("B:%s:s%d" % (ds, k0))   # rfalse_train.csv
                self.add(Step(
                    key="C:%s:s%d" % (ds, k), letter="C",
                    cmd=self._mod("src.rigr.fit_models",
                                  "--data", self.data_dir(ds), "--seed", k,
                                  "--out", out,
                                  "--scorer_model", c["scorer_model"],
                                  "--val_fraction", c["val_fraction"]),
                    outputs=[os.path.join(out, "scorer.joblib"),
                             os.path.join(out, "report.json")],
                    deps=deps, lane=self._lane("C", ds),
                    note="pair scorer (+ temperature on a subject-disjoint 30% "
                         "slice) and the deployed R_false head"))
            if want_omega0:
                out0 = self.model_dir(ds, k0, "omega0")
                self.add(Step(
                    key="C:%s:s%d:w0" % (ds, k0), letter="C",
                    cmd=self._mod("src.rigr.fit_models",
                                  "--data", self.data_dir(ds, "omega0"),
                                  "--seed", k0, "--out", out0, "--no_rfalse",
                                  "--scorer_model", c["scorer_model"],
                                  "--val_fraction", c["val_fraction"]),
                    outputs=[os.path.join(out0, "scorer.joblib")],
                    deps=["B:%s:s%d:w0" % (ds, k0)], lane=self._lane("C", ds),
                    note="omega = 0 scorer for the Tab.3 ladder"))

    # -------------------------------------------------------- run_rigr args
    def _rigr_cmd(self, ds, seed, mode, out, *, head_ckpt, scorer, evidence="auto",
                  orientation="learned", mu=None, btr_false=None,
                  btr_variant=None, btr_runs_dir=None, sweep=True):
        c = self.cfg
        a = ["--dataset", ds, "--seed", seed, "--mode", mode,
             "--pred_dir", self.pred_dir(ds, seed), "--out", out,
             "--head_ckpt", head_ckpt, "--scorer", scorer,
             "--evidence", evidence, "--orientation", orientation,
             "--lam", c["lam"], "--eta", c["eta"], "--tau", c["tau"],
             "--bio", c["bio"], "--gpu", 0]
        a += self.cap_args("test", ds)
        if mu is not None:
            a += ["--mu", mu]
        if mode == "risk":
            if btr_false:
                a += ["--btr_false", btr_false]
            if btr_variant:
                a += ["--btr_variant", btr_variant]
            if btr_runs_dir:
                a += ["--btr_runs_dir", btr_runs_dir]
        if sweep:
            a += ["--sweep", "--sweep_bio", c["sweep_bio"],
                  "--sweep_lam", c["sweep_lam"], "--sweep_tau", c["sweep_tau"]]
            if ds in (c.get("sweep_scale_work_datasets") or []):
                a += ["--sweep_scale", "work"]
        return self._mod("src.rigr.run_rigr", *a)

    # ------------------------------------------------------------------ D
    def plan_D(self, datasets, seeds):
        c = self.cfg
        for mode in c["modes"]:
            for ds in datasets:
                for k in seeds:
                    out = os.path.join(c["rigr_root"], mode, ds, "seed%d" % k)
                    sw = (c.get("sweep_seeds") is None
                          or k in c["sweep_seeds"])
                    # A non-sweeping run writes no sweep_*.csv, so those must
                    # not stay in its declared outputs -- otherwise the step
                    # could never count DONE and would be re-run for ever.
                    outs = [os.path.join(out, "per_image.csv"),
                            os.path.join(out, "summary.json")]
                    if sw:
                        outs += [os.path.join(out, "sweep_curve.csv"),
                                 os.path.join(out, "sweep_per_image.csv")]
                    self.add(Step(
                        key="D:%s:%s:s%d" % (mode, ds, k), letter="D",
                        cmd=self._rigr_cmd(
                            ds, k, mode, out,
                            head_ckpt=self.head_ckpt(ds, k),
                            scorer=os.path.join(self.model_dir(ds, k),
                                                "scorer.joblib"),
                            btr_false=os.path.join(self.model_dir(ds, k),
                                                   "btr_false_deployed.joblib"),
                            btr_runs_dir=c["btr_runs_dir"],
                            btr_variant=c["btr_variant"],
                            sweep=sw),
                        # ``edges/<image>.csv`` is written too (Fig.5 reads it),
                        # but it is not a completion marker: an image with zero
                        # candidates legitimately produces no edges directory.
                        outputs=outs,
                        deps=["A:%s:s%d" % (ds, k), "C:%s:s%d" % (ds, k)],
                        lane=self._lane("D", ds),
                        note="seed-matched head + scorer on the seed-%d test "
                             "predictions" % k))

    # ------------------------------------------------------------------ E
    def plan_E(self, datasets, seeds):
        c = self.cfg
        ck_dir = os.path.join(c["repair_root"], "ckpt")
        train_seed = c["baseline_train_seed"]
        for ds in datasets:
            if "rnca" in c["baseline_methods"]:
                out = os.path.join(ck_dir, "rnca_%s.pt" % ds)
                self.add(Step(
                    key="E:train:rnca:%s" % ds, letter="E",
                    cmd=self._mod("src.baselines.rnca.rnca_adapter",
                                  "--dataset", ds, "--gpu", 0,
                                  "--epochs", c["rnca_iters"],
                                  "--seed", train_seed, "--out", out,
                                  "--pred_dir", self.oof_dir(ds),
                                  *self.cap_args("train", ds)),
                    outputs=[out], lane=self._lane("E_train", ds),
                    note="trained ONCE per dataset (seed %d) on the same "
                         "training inputs RiGR sees: OOF predicted mask + the "
                         "uniform capsule severances" % train_seed))
            if "evapore" in c["baseline_methods"]:
                out = os.path.join(ck_dir, "evapore_%s.pt" % ds)
                self.add(Step(
                    key="E:train:evapore:%s" % ds, letter="E",
                    cmd=self._mod("src.baselines.evapore.evapore_adapter",
                                  "--dataset", ds, "--gpu", 0,
                                  "--epochs", c["evapore_epochs"],
                                  "--seed", train_seed, "--out", out,
                                  "--pred_dir", self.oof_dir(ds),
                                  *self.cap_args("train", ds)),
                    outputs=[out], lane=self._lane("E_train", ds),
                    note="trained ONCE per dataset (seed %d), same inputs as "
                         "rNCA and RiGR" % train_seed))

        device = "cuda:0" if c["baseline_infer_gpu"] else "cpu"

        # --- S6 baselines: EVAPORE is scored in TWO inference variants
        # from the SINGLE evapore_<ds>.pt checkpoint, because "EVAPORE" is
        # ambiguous as a baseline and the two readings answer different
        # questions:
        #   evapore_e2e    -- their candidate generator + their classifier +
        #                     their per-path threshold acceptance (upstream
        #                     end-to-end; only our FOV/protocol is ours)
        #   evapore_scorer -- their classifier scoring OUR candidates, with our
        #                     greedy one-edge-per-endpoint acceptance (the
        #                     controlled comparison against RiGR: candidate set
        #                     and tube painting held fixed, only the score
        #                     changes)
        # Training is unchanged -- one checkpoint per dataset, keyed "evapore".
        # See src/baselines/evapore/evapore_adapter.py.
        infer_methods: List[str] = []
        for _m in c["baseline_methods"]:
            infer_methods += (["evapore_e2e", "evapore_scorer"]
                              if _m == "evapore" else [_m])

        def _ckpt_key(m: str) -> str:
            return "evapore" if m.startswith("evapore") else m
        # --- end S6 baselines edit

        for m in infer_methods:
            for ds in datasets:
                for k in seeds:
                    out = os.path.join(c["repair_root"], m, ds, "seed%d" % k)
                    a = ["--method", m, "--dataset", ds,
                         "--pred_dir", self.pred_dir(ds, k), "--out", out,
                         "--seed", k, "--bio", c["bio"]]
                    a += self.cap_args("test", ds)
                    deps: List[str] = []
                    if _ckpt_key(m) in ("rnca", "evapore"):
                        a += ["--ckpt", os.path.join(
                                  ck_dir, "%s_%s.pt" % (_ckpt_key(m), ds)),
                              "--device", device]
                        deps.append("E:train:%s:%s" % (_ckpt_key(m), ds))
                    self.add(Step(
                        key="E:%s:%s:s%d" % (m, ds, k), letter="E",
                        cmd=self._mod("src.baselines.run_baseline", *a),
                        outputs=[os.path.join(out, "per_image.csv")],
                        deps=deps,
                        lane=(self.lane_of(ds) if c["baseline_infer_gpu"]
                              and m != "geometric" else "cpu"),
                        note="same seed-matched test predictions as RiGR"))

    # ------------------------------------------------------------------ F
    def plan_F(self, datasets, seeds):
        c = self.cfg
        k0 = min(seeds)
        for name, mode, evidence, orientation, mu, data in ABLATIONS:
            for ds in datasets:
                out = os.path.join(c["rigr_root"], "ablation_" + name, ds,
                                   "seed%d" % k0)
                mdir = self.model_dir(ds, k0, "main" if data == "main" else "omega0")
                deps = ["A:%s:s%d" % (ds, k0)]
                deps.append("C:%s:s%d" % (ds, k0) if data == "main"
                            else "C:%s:s%d:w0" % (ds, k0))
                self.add(Step(
                    key="F:%s:%s" % (name, ds), letter="F",
                    cmd=self._rigr_cmd(
                        ds, k0, mode, out,
                        head_ckpt=self.head_ckpt(ds, k0),
                        scorer=os.path.join(mdir, "scorer.joblib"),
                        evidence=evidence, orientation=orientation, mu=mu,
                        btr_false=os.path.join(self.model_dir(ds, k0),
                                               "btr_false_deployed.joblib"),
                        btr_runs_dir=c["btr_runs_dir"],
                        btr_variant=c["btr_variant"],
                        sweep=False),
                    outputs=[os.path.join(out, "per_image.csv")],
                    deps=deps, lane=self._lane("F", ds),
                    note="Tab.3 %s: mode=%s evidence=%s q=%s mu=%s data=%s"
                         % (name, mode, evidence, orientation,
                            "default" if mu is None else mu, data)))

    # ------------------------------------------------------------------ G
    def plan_G(self, datasets, seeds):
        c = self.cfg
        k0 = min(seeds)
        all_ds = list(c["datasets"])
        for held in datasets:
            src = [d for d in all_ds if d != held]
            if len(src) < 2:
                continue
            tag = "lodo_%s" % held
            hdir = os.path.join(c["head_root"], tag, "seed%d" % k0)
            ddir = os.path.join(c["data_root"], tag)
            mdir = os.path.join(c["model_root"], tag, "seed%d" % k0)

            self.add(Step(
                key="G:head:%s" % held, letter="G",
                cmd=self._mod("src.rigr.train_head",
                              "--datasets", ",".join(src), "--tag", tag,
                              "--seed", k0, "--gpu", 0,
                              "--epochs", c["lodo_head_epochs"],
                              "--iters", c["head_iters"],
                              "--workers", c["head_workers"],
                              "--threads", c["head_threads"],
                              "--pred_root", c["oof_root"], "--out", hdir,
                              *(["--limit", str(max(self.cap("train", d) or 0
                                                    for d in src))]
                                if any(self.cap("train", d) for d in src)
                                else [])),
                # ``history.json`` is written ONCE, after the last epoch
                # (train_head.py), while ``best.pt`` is rewritten on every
                # validation improvement -- so best.pt alone appears minutes
                # into a run.  On 2026-09-03 an fd exhaustion left three LODO
                # heads with a 1-epoch best.pt and no history.json, and a
                # waiting chain read that as DONE and started G:data on it.
                # Requiring both makes a partial head impossible to mistake
                # for a finished one.
                outputs=[os.path.join(hdir, "best.pt"),
                         os.path.join(hdir, "history.json")],
                deps=[], lane=self._lane("G_head", held),
                note="head trained on %s only (never %s)" % ("+".join(src), held)))

            self.add(Step(
                key="G:data:%s" % held, letter="G",
                cmd=self._mod("src.rigr.build_data",
                              "--datasets", ",".join(src), "--out", ddir,
                              "--seed", k0, "--oof_root", c["oof_root"],
                              "--head_ckpt", os.path.join(hdir, "best.pt"),
                              "--gpu", 0, "--omega", c["omega"],
                              "--n_cuts", c["n_cuts"], "--rfalse",
                              "--bio", c["rfalse_bio"],
                              "--max_cand_bio", c["max_cand_bio"],
                              "--workers", c["rfalse_workers"],
                              "--fd_rotations", c["rfalse_fd_rotations"],
                              *(["--rfalse_max_images",
                                 ",".join("%s=%d" % (d, n) for d, n
                                          in sorted((c.get("rfalse_max_images")
                                                     or {}).items()) if d in src)]
                                if any(d in (c.get("rfalse_max_images") or {})
                                       for d in src) else []),
                              *(["--per_dataset_limit",
                                 ",".join("%s=%d" % (d, self.cap("train", d))
                                          for d in src if self.cap("train", d))]
                                if any(self.cap("train", d) for d in src) else [])),
                outputs=[os.path.join(ddir, "seed%d" % k0, "pairs.npz"),
                         os.path.join(ddir, "rfalse_train.csv")],
                deps=["G:head:%s" % held], lane=self._lane("G_data", held),
                note="pi + pairs + deployed R_false from the three source "
                     "domains only"))

            self.add(Step(
                key="G:fit:%s" % held, letter="G",
                cmd=self._mod("src.rigr.fit_models",
                              "--data", ddir, "--seed", k0, "--out", mdir,
                              "--scorer_model", c["scorer_model"],
                              "--val_fraction", c["val_fraction"]),
                outputs=[os.path.join(mdir, "scorer.joblib"),
                         os.path.join(mdir, "report.json")],
                deps=["G:data:%s" % held], lane=self._lane("G", held)))

            btr_variant = "wo_%s" % C1_DATASET_NAMES.get(held, held.upper())
            btr_runs_dir = c.get("lodo_btr_runs_dir") or None
            for mode in c["lodo_modes"]:
                out = os.path.join(c["rigr_root"], tag, mode)
                self.add(Step(
                    key="G:run:%s:%s" % (held, mode), letter="G",
                    cmd=self._rigr_cmd(
                        held, k0, mode, out,
                        head_ckpt=os.path.join(hdir, "best.pt"),
                        scorer=os.path.join(mdir, "scorer.joblib"),
                        btr_false=(os.path.join(mdir, "btr_false_deployed.joblib")
                                   if c.get("lodo_btr_deployed_false") else None),
                        btr_variant=btr_variant, btr_runs_dir=btr_runs_dir,
                        sweep=False),
                    outputs=[os.path.join(out, "per_image.csv")],
                    deps=["G:fit:%s" % held], lane=self._lane("G", held),
                    note="R_miss/R_false from %s/btr_*_%s.joblib when they "
                         "exist; run_rigr warns and falls back to the all-data "
                         "head otherwise"
                         % (btr_runs_dir or "runs/btr", btr_variant)))

    # ------------------------------------------------------------------ H
    def plan_H(self, datasets, seeds):
        c = self.cfg
        k0 = min(seeds)
        out = os.path.join(c["results_dir"], "tab1_exp2.csv")
        deps = []
        for ds in datasets:
            deps += ["A:%s:s%d" % (ds, k0), "C:%s:s%d" % (ds, k0)]
        self.add(Step(
            key="H:exp2", letter="H",
            cmd=self._mod("src.rigr.exp2_candidates",
                          "--datasets", ",".join(datasets), "--seed", k0,
                          "--pred_root", c["seg_root"],
                          "--head_root", c["head_root"],
                          "--model_root", c["model_root"],
                          "--bio", c["exp2_bio"],
                          "--max_cand", c["exp2_max_cand"],
                          "--workers", c["exp2_workers"],
                          # DECISIONS.md 2026-09-03 08:30 / 09:1x(c): the
                          # deployed R_miss must come from habs_trainonly, the
                          # same directory D and F use.  Without these two args
                          # exp2 resolved R_miss from the flat runs/btr layout
                          # (H_dep, fitted on test-image events) and BTRBackend
                          # refused to pair it with the habs deployed R_false.
                          "--btr_runs_dir", c["btr_runs_dir"],
                          "--btr_variant", c["btr_variant"],
                          "--gpu", -1, "--out", out,
                          *(["--limit",
                             ",".join("%s=%d" % (d, self.cap("test", d))
                                      for d in datasets if self.cap("test", d))]
                            if any(self.cap("test", d) for d in datasets) else [])),
            outputs=[out], deps=deps, lane="cpu",
            note="applies every candidate edge on its own and measures the "
                 "realised biomarker harm (Tab.1 lower half)"))

    # ------------------------------------------------------------------ I
    def plan_I(self, datasets, seeds):
        c = self.cfg
        k0 = int(c["robust_seed"])
        res = c["results_dir"]
        for ds in datasets:
            for repair in ("none", "rigr"):
                mode = "none" if repair == "none" else "rigr_cmd"
                a = ["--dataset", ds, "--ckpt", self.seg_ckpt(ds, k0),
                     "--gpu", 0, "--repair", mode, "--K", c["stability_K"],
                     "--seed", k0, "--out-dir", res]
                a += self.cap_args("test", ds)
                env = {}
                if repair == "rigr":
                    a += ["--repair-fn", "src.rigr.repair_hook:repair"]
                    env = self._hook_env(ds, k0)
                self.add(Step(
                    key="I:stab:%s:%s" % (ds, repair), letter="I",
                    cmd=self._mod("src.robust.run_stability", *a),
                    outputs=[os.path.join(
                        res, "fig4a_stability_%s_%s_per_image.csv" % (ds, mode))],
                    deps=(["A:%s:s%d" % (ds, k0), "C:%s:s%d" % (ds, k0)]
                          if repair == "rigr" else []),
                    lane=self._lane("I", ds), env=env,
                    note="Fig.4A %s" % ("baseline" if repair == "none"
                                        else "risk-guided RiGR repair")))

        rds = c["resolution_dataset"]
        if rds in datasets:
            for repair in ("none", "rigr"):
                tag = "" if repair == "none" else "_rigr"
                a = ["--dataset", rds, "--ckpt", self.seg_ckpt(rds, k0),
                     "--gpu", 0, "--out-dir", res, "--tag", tag]
                a += self.cap_args("test", rds)
                env = {}
                if repair == "rigr":
                    a += ["--repair", "rigr_cmd",
                          "--repair-fn", "src.rigr.repair_hook:repair"]
                    env = self._hook_env(rds, k0)
                self.add(Step(
                    key="I:res:%s:%s" % (rds, repair), letter="I",
                    cmd=self._mod("src.robust.run_resolution", *a),
                    outputs=[os.path.join(
                        res, "fig4b_resolution_%s%s_drift.csv" % (rds, tag))],
                    deps=(["A:%s:s%d" % (rds, k0), "C:%s:s%d" % (rds, k0)]
                          if repair == "rigr" else []),
                    lane=self._lane("I", rds), env=env,
                    note="Fig.4B %s" % ("baseline" if repair == "none"
                                        else "risk-guided RiGR repair")))

    def _hook_env(self, ds: str, seed: int) -> Dict[str, str]:
        c = self.cfg
        m = self.model_dir(ds, seed)
        return {
            "RIGR_HOOK_HEAD": self.head_ckpt(ds, seed),
            "RIGR_HOOK_SCORER": os.path.join(m, "scorer.joblib"),
            "RIGR_HOOK_MODE": "risk",
            "RIGR_HOOK_GPU": "0",
            "RIGR_HOOK_LAM": str(c["lam"]),
            "RIGR_HOOK_ETA": str(c["eta"]),
            "RIGR_HOOK_TAU": str(c["tau"]),
            "RIGR_HOOK_BTR_FALSE": os.path.join(m, "btr_false_deployed.joblib"),
            # DECISIONS.md 2026-09-06 12:30/12:40: repair_hook resolves the
            # deployed R_miss head via RIGR_HOOK_BTR_RUNS_DIR (habs_trainonly,
            # same dir D/F/H use) + RIGR_HOOK_BTR_VARIANT; without them the
            # 09-06 Fig.4 RiGR-arm runs errored on every image (miss/false
            # head target mismatch).
            "RIGR_HOOK_BTR_RUNS_DIR": c["btr_runs_dir"],
            "RIGR_HOOK_BTR_VARIANT": c["btr_variant"],
        }

    # ------------------------------------------------------------------ J
    def plan_J(self, prior_keys: Sequence[str]):
        c = self.cfg
        self.add(Step(
            key="J:make_all", letter="J",
            cmd=self._mod("src.eval.make_all"),
            outputs=[os.path.join(c["results_dir"], "tab2.md"),
                     os.path.join(c["results_dir"], "tab3.md")],
            deps=list(prior_keys), lane="cpu",
            note="every table and figure; each build_* skips what is missing"))

    # ------------------------------------------------------------------
    def apply_skips(self, globs: Sequence[str]) -> List[str]:
        """Mark every step whose key matches one of ``globs`` as skipped."""
        hit: List[str] = []
        for k, st in self.steps.items():
            if any(fnmatch.fnmatch(k, g) for g in globs):
                st.skip = True
                hit.append(k)
        return sorted(hit)

    def build(self, letters: Sequence[str], datasets: Sequence[str],
              seeds: Sequence[int]) -> "Dict[str, Step]":
        want = set(letters)
        want_omega0 = "F" in want
        if "A" in want:
            self.plan_A(datasets, seeds)
        if "B" in want:
            self.plan_B(datasets, seeds, want_omega0)
        if "C" in want:
            self.plan_C(datasets, seeds, want_omega0)
        if "D" in want:
            self.plan_D(datasets, seeds)
        if "E" in want:
            self.plan_E(datasets, seeds)
        if "F" in want:
            self.plan_F(datasets, seeds)
        if "G" in want:
            self.plan_G(datasets, seeds)
        if "H" in want:
            self.plan_H(datasets, seeds)
        if "I" in want:
            self.plan_I(datasets, seeds)
        if "J" in want:
            self.plan_J([k for k in self.steps])
        # dependencies on steps that were filtered out are simply dropped:
        # they are then assumed already done on disk (which --status shows).
        known = set(self.steps)
        for st in self.steps.values():
            st.deps = [d for d in st.deps if d in known]
        return self.steps


# --------------------------------------------------------------------------
def topo_order(steps: Dict[str, Step]) -> List[str]:
    seen: Dict[str, int] = {}
    order: List[str] = []

    def visit(k: str, stack: Sequence[str] = ()):
        if seen.get(k) == 2:
            return
        if seen.get(k) == 1:
            raise ValueError("dependency cycle: %s" % " -> ".join(list(stack) + [k]))
        seen[k] = 1
        for d in steps[k].deps:
            visit(d, list(stack) + [k])
        seen[k] = 2
        order.append(k)

    for k in steps:
        visit(k)
    return order


# --------------------------------------------------------------------------
class Runner:
    """Dependency-aware scheduler over one gpu0, one gpu1 and N cpu slots."""

    def __init__(self, steps: Dict[str, Step], cfg: Dict[str, Any],
                 force: bool = False, verbose: bool = True):
        self.steps = steps
        self.cfg = cfg
        self.force = force
        self.verbose = verbose
        self.state: Dict[str, str] = {}
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.capacity = {"gpu0": 1, "gpu1": 1, "cpu": max(1, int(cfg["cpu_jobs"]))}
        self.running = {"gpu0": 0, "gpu1": 0, "cpu": 0}
        self.log_dir = os.path.join(EXP_ROOT, cfg["log_dir"])
        # A hard TerminateProcess (taskkill /F, a harness stop-all) cannot be
        # caught and loses whatever is still in stdout buffers -- which is
        # exactly how the 2026-09-03 09:56 mass kill left a log ending mid-run
        # with no summary and no traceback.  This file is rewritten after
        # every state change, so a postmortem always knows what was running.
        # per-pid: two orchestrators can legitimately run at once (a
        # recovery pass alongside a chain stage) and must not overwrite
        # each other postmortem.  _state.json stays as a symlink-free
        # copy of the most recent writer for convenience.
        self.state_path = os.path.join(EXP_ROOT, cfg["log_dir"],
                                       "_state_%d.json" % os.getpid())
        os.makedirs(self.log_dir, exist_ok=True)

    def _dump_state(self, note: str = "") -> None:
        try:
            tmp = self.state_path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(dict(pid=os.getpid(), note=note,
                               updated=time.strftime("%Y-%m-%d %H:%M:%S"),
                               running=[k for k, v in self.state.items()
                                        if v == "RUNNING"],
                               state=dict(self.state)), fh, indent=1)
            os.replace(tmp, self.state_path)
            import shutil
            shutil.copyfile(self.state_path,
                            os.path.join(os.path.dirname(self.state_path),
                                         "_state.json"))
        except Exception:
            pass

    # -- one step --------------------------------------------------------
    def _env_for(self, step: Step) -> Dict[str, str]:
        env = dict(os.environ)
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONPATH"] = EXP_ROOT + os.pathsep + env.get("PYTHONPATH", "")
        if step.lane == "gpu0":
            env["CUDA_VISIBLE_DEVICES"] = "0"
        elif step.lane == "gpu1":
            env["CUDA_VISIBLE_DEVICES"] = "1"
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        env.update({k: str(v) for k, v in step.env.items()})
        return env

    def _run_one(self, key: str) -> None:
        step = self.steps[key]
        log = os.path.join(self.log_dir, key.replace(":", "_") + ".log")
        t0 = time.time()
        print("[s4][%s] START  lane=%s  -> %s" % (key, step.lane, log), flush=True)
        rc = 1
        try:
            with open(log, "w", encoding="utf-8", errors="replace") as fh:
                fh.write("# %s\n# lane=%s\n# %s\n\n"
                         % (key, step.lane, shell(step)))
                fh.flush()
                rc = subprocess.call(step.cmd, cwd=EXP_ROOT,
                                     env=self._env_for(step),
                                     stdout=fh, stderr=subprocess.STDOUT)
        except Exception as exc:  # noqa: BLE001
            print("[s4][%s] launch failed: %s" % (key, exc), flush=True)
        dt = time.time() - t0
        missing = step.missing()
        ok = (rc == 0) and not missing
        with self.cv:
            self.state[key] = "DONE" if ok else "FAILED"
            self._dump_state("finished " + key)
            self.running[step.lane] -= 1
            self.cv.notify_all()
        if ok:
            print("[s4][%s] DONE   %.1fs" % (key, dt), flush=True)
        else:
            print("[s4][%s] FAILED rc=%s %.1fs  missing=%s  (see %s)"
                  % (key, rc, dt, missing[:3], log), flush=True)

    # -- scheduler -------------------------------------------------------
    def run(self) -> int:
        order = topo_order(self.steps)
        for k in order:
            st = self.steps[k]
            if st.done() and not (self.force and not st.skip):
                # a skipped step whose outputs exist is DONE like any other --
                # that is how a remotely-produced checkpoint gets picked up
                self.state[k] = "DONE"
            elif st.skip:
                self.state[k] = "SKIPPED"
            else:
                self.state[k] = "PENDING"
        n_done = sum(1 for k in order if self.state[k] == "DONE")
        n_skipped = sum(1 for k in order if self.state[k] == "SKIPPED")
        if n_done:
            print("[s4] %d/%d steps already complete (outputs exist)"
                  % (n_done, len(order)), flush=True)
        if n_skipped:
            print("[s4] %d step(s) skipped by --skip-steps and not yet on disk: "
                  "%s" % (n_skipped, ", ".join(k for k in order
                                               if self.state[k] == "SKIPPED")),
                  flush=True)

        threads: List[threading.Thread] = []
        with self.cv:
            while True:
                if all(self.state[k] in ("DONE", "FAILED", "BLOCKED", "SKIPPED")
                       for k in order):
                    break
                launched = False
                for k in order:
                    if self.state[k] != "PENDING":
                        continue
                    dep_states = [self.state[d] for d in self.steps[k].deps]
                    if any(s in ("FAILED", "BLOCKED") for s in dep_states):
                        self.state[k] = "BLOCKED"
                        print("[s4][%s] BLOCKED (a dependency failed)" % k,
                              flush=True)
                        launched = True
                        continue
                    if any(s != "DONE" for s in dep_states):
                        # includes a SKIPPED dependency whose output has not
                        # arrived: leave this step PENDING (a later invocation
                        # will pick it up), never BLOCKED
                        continue
                    lane = self.steps[k].lane
                    if self.running[lane] >= self.capacity[lane]:
                        continue
                    self.state[k] = "RUNNING"
                    self._dump_state("launched " + k)
                    self.running[lane] += 1
                    t = threading.Thread(target=self._run_one, args=(k,),
                                         daemon=True)
                    t.start()
                    threads.append(t)
                    launched = True
                if not launched:
                    if not any(s == "RUNNING" for s in self.state.values()):
                        break
                    self.cv.wait(timeout=5.0)
        for t in threads:
            t.join()

        # anything still PENDING was waiting on a --skip-steps dependency whose
        # output never arrived; that is a deferral, not a failure
        for k in order:
            if self.state[k] == "PENDING":
                self.state[k] = "DEFERRED"
        n_done = sum(1 for k in order if self.state[k] == "DONE")
        n_fail = sum(1 for k in order if self.state[k] == "FAILED")
        n_block = sum(1 for k in order if self.state[k] == "BLOCKED")
        n_skipped = sum(1 for k in order if self.state[k] == "SKIPPED")
        n_defer = sum(1 for k in order if self.state[k] == "DEFERRED")
        print("\n%s SUMMARY %s" % ("=" * 22, "=" * 22), flush=True)
        for k in order:
            print("  %-9s %s" % (self.state[k], k))
        print("\n%d done, %d failed, %d blocked, %d skipped, %d deferred "
              "(of %d)" % (n_done, n_fail, n_block, n_skipped, n_defer,
                           len(order)), flush=True)
        self._dump_state("summary")
        if n_skipped or n_defer:
            print("re-run the same command once the skipped steps' outputs "
                  "exist to pick them (and their dependants) up.")
        return 1 if (n_fail or n_block) else 0


# --------------------------------------------------------------------------
def load_config(path: Optional[str]) -> Dict[str, Any]:
    cfg = json.loads(json.dumps(DEFAULTS))       # deep copy
    cfg["python"] = DEFAULTS["python"]
    if path:
        with open(path, "r", encoding="utf-8") as f:
            user = json.load(f)
        unknown = sorted(set(user) - set(cfg))
        if unknown:
            raise SystemExit("unknown config keys: %s" % unknown)
        cfg.update(user)
    return cfg


def _mtime(path: str) -> str:
    """``mtime`` of a file as a log-friendly string (``?`` when absent).

    Printed for every BTR head the preflight touches so a finished table can
    always be traced back to *which build* of the heads priced it -- the C1
    agent refits them in place, so the path alone does not identify a version.
    """
    try:
        t = os.path.getmtime(path)
    except OSError:
        return "?"
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))


def preflight_sigma(datasets) -> List[str]:
    """Check the training-split Gate A sigma table actually covers the run.

    sigma is the denominator of every ``macro_mae_*`` number in the paper, so
    two failure modes have to be caught before a phase starts rather than at
    aggregation time:

    * a **dataset with no rows** -- ``load_gt_scales`` warns and returns {},
      and every biomarker error for that dataset silently reads NaN;
    * a row measured at a **non-native resolution** -- predictions are mapped
      back to native before the biomarker call (see
      ``src.eval.biomarker_eval``), so a sigma estimated at the C1 working
      scale is simply the wrong denominator, and nothing downstream can tell.
    """
    import pandas as pd

    from src.eval.biomarker_eval import GATE_A_SCALES_CSV, PRIMARY
    from src.seg.data import canon

    if not os.path.exists(GATE_A_SCALES_CSV):
        return ["missing %s (the training-split sigma table); every step that "
                "computes a biomarker error will raise"
                % os.path.relpath(GATE_A_SCALES_CSV, EXP_ROOT)]
    df = pd.read_csv(GATE_A_SCALES_CSV)
    df["_ds"] = df["dataset"].astype(str).map(canon)
    need = {"%s_%s" % (b, p) for b in PRIMARY for p in ("pvbm", "skan")}
    msgs: List[str] = []
    for ds in datasets:
        sub = df[df["_ds"] == canon(ds)]
        if sub.empty:
            msgs.append("%s has NO rows in %s -- every macro_mae_* value for "
                        "it would be NaN. The C1 agent owns this file."
                        % (ds, os.path.relpath(GATE_A_SCALES_CSV, EXP_ROOT)))
            continue
        miss = sorted(need - set(sub["biomarker"].astype(str)))
        if miss:
            msgs.append("%s is missing sigma for %s" % (ds, miss))
        if "resolution" in sub.columns:
            bad = sorted({str(r) for r in sub["resolution"] if str(r) != "native"})
            if bad:
                msgs.append("%s sigma was measured at %s, not native -- the "
                            "biomarker contract maps every prediction back to "
                            "native resolution first, so this is the wrong "
                            "denominator for that dataset" % (ds, ", ".join(bad)))
    return msgs


def preflight_btr(cfg: Dict[str, Any], letters: Sequence[str], datasets,
                  seeds) -> List[str]:
    """Load the risk backend D/F will use, and refuse to schedule on mismatch.

    ``U(e) = p R_miss - lambda (1-p) R_false - eta C_geom`` only means anything
    if both heads price the same quantity.  Three ways that silently breaks,
    all of which are checked here rather than discovered in a finished table:

    1. the two heads were fitted on different target families (``H_dep`` vs
       ``H_abs``) -- :class:`~src.rigr.btr_backend.BTRBackend` raises, and
       ``run_rigr`` used to catch only ``FileNotFoundError``;
    2. a head's scalarisation prefix does not match its own stored targets, in
       which case ``BTRHead.predict`` returns NaN and
       ``src.rigr.utility.risk_values`` turns that into ``R = 1`` -- a
       uniform-cost run still labelled ``risk``;
    3. the R_miss head is the flat ``runs/btr/btr_*_all.joblib`` one, which was
       fitted on perturbations of the very test images RiGR is applied to.

    Returns a list of fatal messages (empty when everything checks out).
    """
    if not ({"D", "F"} & set(letters)):
        return []
    import numpy as _np

    fatal: List[str] = []
    miss_path = os.path.join(EXP_ROOT, cfg["btr_runs_dir"],
                             "btr_miss_%s.joblib" % cfg["btr_variant"])
    if not os.path.exists(miss_path):
        return ["D/F need the deployed R_miss head %s (DECISIONS.md 2026-09-03 "
                "08:30: H_abs, training-split only). Run `python -m src.c1.btr "
                "--target habs --train-split-only` or copy it in."
                % os.path.relpath(miss_path, EXP_ROOT)]

    from src.c1.perturb import PHI_COLUMNS
    from src.rigr.btr_backend import BTRBackend

    print("[s4][preflight] R_miss head %s  mtime=%s"
          % (os.path.relpath(miss_path, EXP_ROOT), _mtime(miss_path)))

    try:
        import pandas as pd

        probe = pd.DataFrame(_np.zeros((4, len(PHI_COLUMNS))), columns=PHI_COLUMNS)
    except Exception as exc:                                  # noqa: BLE001
        return ["preflight could not build a probe frame: %s" % exc]

    seen_false = 0
    k0 = min(seeds)
    wanted = sorted({(ds, k) for ds in datasets
                     for k in (seeds if "D" in letters else [k0])})
    for ds, k in wanted:
        false_path = os.path.join(EXP_ROOT, cfg["model_root"], ds,
                                  "seed%d" % k, "btr_false_deployed.joblib")
        if not os.path.exists(false_path):
            continue
        try:
            be = BTRBackend(miss_path, false_path)
        except ValueError as exc:                             # target mismatch
            fatal.append(
                "%s seed%d: %s -- refit step C with "
                "`python -m src.rigr.fit_models --data runs/rigr_data/%s "
                "--seed %d --out runs/rigr_models/%s/seed%d` after rebuilding "
                "step B with --force (build_data now stores Habs_* as well)."
                % (ds, k, exc, ds, k, ds, k))
            continue
        except Exception as exc:                              # noqa: BLE001
            fatal.append("%s seed%d: could not load the risk backend (%s: %s)"
                         % (ds, k, type(exc).__name__, exc))
            continue
        print("[s4][preflight] %-9s seed%d R_false %s  mtime=%s  target=%s"
              % (ds, k, os.path.relpath(false_path, EXP_ROOT),
                 _mtime(false_path), be.target))
        n_before = len(fatal)
        for kind, fn in (("R_miss", be.predict_R_miss),
                         ("R_false", be.predict_R_false)):
            try:
                v = _np.asarray(fn(probe), dtype=float).ravel()
            except Exception as exc:                          # noqa: BLE001
                fatal.append("%s seed%d: %s raised %s: %s"
                             % (ds, k, kind, type(exc).__name__, exc))
                continue
            if not _np.isfinite(v).any():
                fatal.append(
                    "%s seed%d: %s returns all-NaN (head %r, targets %s). "
                    "src.rigr.utility.risk_values would silently read that as "
                    "R = 1, i.e. a uniform-cost run reporting risk_backend=%s."
                    % (ds, k, kind,
                       (be._miss if kind == "R_miss" else be._false).name,
                       (be._miss if kind == "R_miss" else be._false).targets[:3],
                       be.__name__))
        if len(fatal) == n_before:
            seen_false += 1
    if seen_false == 0:
        print("[s4][preflight] no deployed R_false head exists yet (step C has "
              "not run for the selected datasets/seeds); the miss head at %s "
              "loads and scalarises, and each D/F step re-checks its own pair "
              "at launch." % os.path.relpath(miss_path, EXP_ROOT))
    else:
        print("[s4][preflight] risk backend OK for %d (dataset, seed) pair(s): "
              "R_miss=%s target=%s"
              % (seen_false, os.path.relpath(miss_path, EXP_ROOT),
                 cfg["btr_target"]))
    return fatal


def _check_environment(cfg: Dict[str, Any], datasets, seeds) -> List[str]:
    """Preflight: the inputs S4 consumes but does not produce."""
    warn: List[str] = []
    for ds in datasets:
        d = os.path.join(EXP_ROOT, cfg["oof_root"], ds, "pred")
        if not os.path.isdir(d):
            warn.append("missing OOF predictions %s (run run_crossfit_all.ps1); "
                        "steps A/B/E for %s cannot run correctly"
                        % (os.path.relpath(d, EXP_ROOT), ds))
        for k in seeds:
            p = os.path.join(EXP_ROOT, cfg["seg_root"], ds, "seed%d" % k, "pred")
            if not os.path.isdir(p):
                warn.append("missing test predictions %s (S2 grid)"
                            % os.path.relpath(p, EXP_ROOT))
    for kind in ("miss", "false"):
        p = os.path.join(EXP_ROOT, cfg["btr_root"], "btr_%s_all.joblib" % kind)
        if not os.path.exists(p):
            warn.append("missing BTR head %s: risk mode will fall back to "
                        "R = 1 and every row will say risk_backend="
                        "fallback-uniform" % os.path.relpath(p, EXP_ROOT))
    lodo_dir = cfg.get("lodo_btr_runs_dir") or cfg["btr_root"]
    for ds in datasets:
        p = os.path.join(EXP_ROOT, lodo_dir, "btr_miss_wo_%s.joblib"
                         % C1_DATASET_NAMES.get(ds, ds.upper()))
        if not os.path.exists(p):
            warn.append("no leave-one-dataset-out BTR head for %s (%s); the "
                        "LODO block will fall back to the all-data head"
                        % (ds, os.path.relpath(p, EXP_ROOT)))
    from src.eval.biomarker_eval import GATE_A_SCALES_CSV

    if not os.path.exists(GATE_A_SCALES_CSV):
        warn.append("missing %s -- DECISIONS.md 2026-09-03 10:20 requires the "
                    "TRAINING-SPLIT sigma table; src.eval.biomarker_eval now "
                    "RAISES rather than silently reporting NaN, so every step "
                    "that computes a biomarker error will fail until it exists"
                    % os.path.relpath(GATE_A_SCALES_CSV, EXP_ROOT))
    return warn


def main(argv=None) -> int:
    # Redirected stdout is block-buffered by default, so a killed run loses
    # everything still in the buffer (that is why the 2026-09-03 09:56 kill
    # left no SUMMARY).  Line buffering costs nothing here and makes the log
    # a faithful record of how far the run got.
    try:
        sys.stdout.reconfigure(line_buffering=True)
        sys.stderr.reconfigure(line_buffering=True)
    except Exception:
        pass
    ap = argparse.ArgumentParser(
        prog="python -m src.pipeline.s4_all",
        description="stage S4 orchestrator (see exp/S4_RUNBOOK.md)")
    ap.add_argument("--only", default=LETTERS,
                    help="comma- or letter-list of steps to plan, e.g. 'A,B,C' "
                         "or 'DEF' (default: everything)")
    ap.add_argument("--datasets", default=None)
    ap.add_argument("--seeds", default=None)
    ap.add_argument("--config", default=None, help="JSON config overriding the "
                                                   "defaults printed by --show-config")
    ap.add_argument("--jobs", type=int, default=None, help="CPU pool size")
    ap.add_argument("--skip-steps", dest="skip_steps", default=None,
                    help="comma-separated fnmatch globs over step keys that "
                         "this invocation must not launch, e.g. 'E:train:*' "
                         "when the rNCA / EVAPORE checkpoints are being "
                         "produced elsewhere and copied into their S4 output "
                         "paths.  A skipped step still counts as DONE the "
                         "moment its outputs exist; until then its dependants "
                         "stay PENDING (reported DEFERRED), never BLOCKED, so "
                         "a later invocation runs them.")
    ap.add_argument("--force", action="store_true",
                    help="re-run steps whose outputs already exist")
    ap.add_argument("--dry-run", action="store_true",
                    help="print every command in dependency order and exit")
    ap.add_argument("--status", action="store_true",
                    help="print what is done / pending and exit")
    ap.add_argument("--show-config", action="store_true")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.jobs:
        cfg["cpu_jobs"] = int(args.jobs)
    if args.show_config:
        print(json.dumps(cfg, indent=2))
        return 0

    letters = [c for c in args.only.upper().replace(",", "") if c in LETTERS]
    if not letters:
        raise SystemExit("--only selected no valid step letters (%s)" % LETTERS)
    datasets = ([d.strip() for d in args.datasets.split(",") if d.strip()]
                if args.datasets else list(cfg["datasets"]))
    seeds = ([int(s) for s in args.seeds.split(",") if s.strip()]
             if args.seeds else list(cfg["seeds"]))

    skip_globs = ([g.strip() for g in args.skip_steps.split(",") if g.strip()]
                  if args.skip_steps else [])
    planner = Planner(cfg)
    steps = planner.build(letters, datasets, seeds)
    skipped_keys = planner.apply_skips(skip_globs) if skip_globs else []
    if skip_globs:
        print("[s4] --skip-steps %s matched %d step(s): %s"
              % (",".join(skip_globs), len(skipped_keys),
                 ", ".join(skipped_keys) or "(none)"))
    order = topo_order(steps)

    sigma_msgs = preflight_sigma(datasets)
    if sigma_msgs:
        # fatal for the phases that report biomarker errors; a warning for the
        # rest so a plumbing-only invocation is not blocked by it
        hard = bool({"B", "D", "F", "G", "H"} & set(letters))
        print("---- sigma table (%s) ----"
              % ("FATAL" if hard else "warning"))
        for m in sigma_msgs:
            print("  [%s] %s" % ("fatal" if hard else "warn", m))
        if hard:
            print("Fix the sigma table (or re-run with --only on steps that do "
                  "not compute biomarker errors) before scheduling.")
            return 2
        print()

    fatal = preflight_btr(cfg, letters, datasets, seeds)
    if fatal:
        print("---- BTR preflight FAILED (D/F not scheduled) ----")
        for m in fatal:
            print("  [fatal] %s" % m)
        return 2

    warn = _check_environment(cfg, datasets, seeds)
    if warn:
        print("---- preflight warnings ----")
        for w in warn:
            print("  [warn] %s" % w)
        print()

    if args.dry_run or args.status:
        lanes = {"gpu0": 0, "gpu1": 0, "cpu": 0}
        n_done = 0
        print("%-34s %-6s %-5s %s" % ("STEP", "LANE", "STATE", "OUTPUT/COMMAND"))
        print("-" * 100)
        for k in order:
            st = steps[k]
            lanes[st.lane] += 1
            done = st.done()
            n_done += int(done)
            state = "DONE" if done else ("SKIP" if st.skip else "TODO")
            print("%-34s %-6s %-5s %s"
                  % (k, st.lane, state, st.outputs[0] if st.outputs else ""))
            if st.note:
                print("%-34s %-6s %-5s   # %s" % ("", "", "", st.note))
            if st.deps:
                print("%-34s %-6s %-5s   deps: %s"
                      % ("", "", "", ", ".join(st.deps)))
            if args.dry_run:
                print("%-34s %-6s %-5s   $ %s" % ("", "", "", shell(st)))
            print()
        print("-" * 100)
        n_skip = sum(1 for k in order if steps[k].skip and not steps[k].done())
        print("%d steps  (%d done, %d pending, %d skipped)  "
              "lanes: gpu0=%d gpu1=%d cpu=%d"
              % (len(order), n_done, len(order) - n_done - n_skip, n_skip,
                 lanes["gpu0"], lanes["gpu1"], lanes["cpu"]))
        return 0

    print("[s4] %d steps, datasets=%s seeds=%s, cpu pool=%d"
          % (len(order), ",".join(datasets), ",".join(map(str, seeds)),
             cfg["cpu_jobs"]), flush=True)
    runner = Runner(steps, cfg, force=args.force)
    try:
        return runner.run()
    except BaseException:
        # Any escape from the scheduler must leave a traceback AND the
        # partial state in the log -- a silent disappearance is what made
        # the 09:56 incident take a forensic session to explain.
        import traceback

        print("[s4] the orchestrator raised; partial state follows",
              flush=True)
        traceback.print_exc()
        try:
            still = [k for k, v in runner.state.items() if v == "RUNNING"]
            for k in order:
                print("  %-9s %s" % (runner.state.get(k, "?"), k), flush=True)
            print("[s4] %d step process(es) were still running and are now "
                  "orphaned (they keep going; re-run to pick up whatever "
                  "they finish): %s" % (len(still), ", ".join(still)),
                  flush=True)
            runner._dump_state("crashed")
        except Exception:
            pass
        raise


if __name__ == "__main__":
    raise SystemExit(main())
