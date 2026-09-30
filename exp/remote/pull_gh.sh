#!/usr/bin/env bash
# Pull every step-G / step-H artefact from the remote mirror into the exact
# local paths the S4 orchestrator declares as those steps' outputs, so a local
# `--status` sees them as DONE.  Idempotent: tar overwrites in place; nothing
# local outside runs/rigr{,_head,_models,_data}/lodo_*, results/tab1_exp2.csv
# and runs/s4_logs/ is touched.
#
#   bash exp/remote/pull_gh.sh            # pull whatever exists
#
set -u
cd "$(dirname "$0")/.."
SSH="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=20 user@GPU_HOST"
R=/path/to/workdir/medical1/exp
STAMP() { date -Is; }

pull() {   # $1 = label, $2 = remote `tar -c` argument list (shell-expanded there)
  local label="$1" spec="$2"
  echo "[pull] $label $(STAMP)"
  # `tar -c` on a glob that matches nothing exits non-zero; the `ls` guard keeps
  # a not-yet-produced artefact from looking like a transfer failure.
  $SSH "cd $R && ls -d $spec >/dev/null 2>&1 && tar czf - --exclude='__pycache__' ${EXC:-} $spec" \
      2>/dev/null | tar xzf - 2>/dev/null \
    && echo "[pull] $label OK $(STAMP)" \
    || echo "[pull] $label nothing-yet-or-failed $(STAMP)"
}

# G -- the runs themselves (per_image.csv / summary.json / edges/ / mask/)
EXC="--exclude=*_sigma_pre* --exclude=*_bdfix* --exclude=*_ooffix* --exclude=*_failed* --exclude=*_partial* --exclude=*_btr_pre*"
pull "G:runs"        "runs/rigr/lodo_*"
# G -- provenance: the LODO head, scorer and deployed R_false + report.json
pull "G:heads"       "runs/rigr_head/lodo_*"
pull "G:models"      "runs/rigr_models/lodo_*"
# G -- the fitted data dir minus the big pair archives
echo "[pull] G:data $(STAMP)"
$SSH "cd $R && ls -d runs/rigr_data/lodo_* >/dev/null 2>&1 && tar czf - --exclude='*.npz' runs/rigr_data/lodo_*" 2>/dev/null \
  | tar xzf - 2>/dev/null && echo "[pull] G:data OK $(STAMP)" || echo "[pull] G:data nothing-yet $(STAMP)"

# H -- the Exp2 candidate-level table
pull "H:tab1_exp2"   "results/tab1_exp2.csv"

# I -- the robustness CSVs (baseline arms are produced remotely too, and they
# are results/ files that nothing else in this script covered)
pull "I:fig4a"       "results/fig4a_stability_*"
pull "I:fig4b"       "results/fig4b_resolution_*"

# STARE supplement: its RiGR runs live under runs/rigr/<mode>/stare/, its head,
# data and models under the plain (non-lodo) roots
pull "STARE:runs"    "runs/rigr/*/stare"
pull "STARE:head"    "runs/rigr_head/stare"
pull "STARE:models"  "runs/rigr_models/stare"
pull "STARE:oof"     "runs/seg_oof/stare"

# logs (per-step stdout, so a failure is diagnosable locally).  They land in
# runs/s4_logs_remote/ so they never collide with the local orchestrator's own
# runs/s4_logs/.
echo "[pull] logs $(STAMP)"
if $SSH "cd $R && tar czf - runs/s4_logs" | tar xzf - --transform 's|^runs/s4_logs|runs/s4_logs_remote|'; then
  echo "[pull] logs OK $(STAMP)"
else
  echo "[pull] logs FAILED $(STAMP)"
fi

echo "[pull] --- local state ---"
ls -d runs/rigr/lodo_* 2>/dev/null || echo "  (no lodo runs yet)"
ls -la results/tab1_exp2.csv 2>/dev/null || echo "  (no tab1_exp2.csv yet)"
echo "[pull] finished $(STAMP)"
