# The conservative reconnector used as the topology counterfactual

*Self-contained description for the paper agent, extracted from the fallback
manuscript `paper/sections/03_method.tex` (§ "RiGR: risk-guided graph repair",
§ "Candidate generation and attachment ports", § "Corridor-restricted lifted-state
A\*", § "Pair scorer and probability calibration", § "Asymmetric expected utility
and global selection") and cross-checked against the run metadata of the arm the
audit actually uses, `exp/runs/rigr/uniform/<dataset>/seed<S>/summary.json`.*

**Purpose of this file.** Reviewer point 20 asks that the conservative
reconnector be described self-containedly in the manuscript, with no forward
reference to a companion paper and no TODO. Everything the audit needs is below.
Nothing here depends on the companion paper's contribution (the learned risk
model): the audit deliberately uses the *degenerate, uniform-event-cost* variant,
which is the arm in which the risk heads are inert.

---

## 1. What it is, in one paragraph

The topology counterfactual is a **post-hoc graph-repair stage applied to the
frozen predicted mask**. It never retrains, never sees the reference mask, and
never modifies the segmenter. It skeletonises the predicted mask, enumerates a
bounded set of candidate reconnections between dangling skeleton endpoints,
scores each candidate with a small calibrated classifier, discards every
candidate whose expected utility is not positive, resolves spatial conflicts,
solves an exact maximum-weight matching under a capacity of one reconnection per
endpoint, and paints the accepted paths. It is *conservative* in three specific
senses, all of which are properties of the configuration used here rather than
of the method in general: (i) it can only **add** vessel pixels along a
geometrically admissible path, never delete them; (ii) each skeleton endpoint may
participate in **at most one** accepted reconnection; and (iii) a candidate is
accepted only when the calibrated probability that the reconnection is one the
reference supports outweighs the penalty for a wrong repair, so the default
operating point accepts a minority of the candidates it generates (measured on
HRF: a median of 209 candidates per image, 127 with positive utility, 56
accepted after conflict pruning and matching).

## 2. Exactly which arm the audit uses

`runs/rigr/uniform/…`, i.e. the **uniform-event-cost** degenerate variant, with
`risk_backend = "uniform"`. In this arm the per-event risk terms are constant, so
the acceptance rule reduces to

  U(e) = p_e − λ (1 − p_e) − η · C̃_geom(e),   accept iff U(e) > 0,

with the run-recorded settings **λ = 1.0, η = 0.1, μ = 2.0, λ_θ = 1.0**, learned
orientation evidence, and a per-dataset geometric-cost scale `geom_scale` read
from `runs/rigr_models/<ds>/seed<S>/geom_scale.json`. There is **no probability
threshold τ in this arm** — τ belongs only to the probability-only baseline. The
audit reports the arm at its default operating point, across segmentation seeds
0–2, on the held-out test split.

## 3. The five stages

### 3.1 Candidate generation
Skeletonise the predicted mask M̂ → Ŝ. Degree-1 skeleton points that are not on
the FOV border band are *dangling endpoints* V_end. An endpoint–endpoint
candidate (i, j) is admitted only if it satisfies **all four** of:

1. **proximity** — normalised separation d_ij / r̄_ij ≤ α, default α = 6, where
   r̄_ij is the mean local vessel radius of the two endpoints;
2. **tangent compatibility** — cos∠(θ_i, u_ij) ≥ τ₁ and cos∠(θ_j, −u_ij) ≥ τ₁
   with τ₁ = 0.5, i.e. both endpoints point roughly along the connecting
   direction;
3. **calibre compatibility** — |log(r_i / r_j)| ≤ log 2, so the two ends differ
   in radius by at most a factor of two;
4. **no interposed vessel** — no vessel disconnected from both endpoints crosses
   the connecting segment roughly perpendicularly.

Endpoint-to-vessel-body (T-shaped) candidates are supported by clustering the
legal attachment positions along a trunk into **attachment-port pseudo-nodes**,
each of capacity 1. The candidate node set is V_c = V_end ∪ V_port, and both
candidate kinds are ordinary unit-capacity edges.

### 3.2 Evidence: a backbone-decoupled micro head
A ≈0.3 M-parameter multi-task head takes the fundus image and the segmenter's
probability map [I, P] and emits (a) a **vessel-evidence** map V_I(x) with
full-pixel supervision, and (b) a **conditional orientation** distribution
q(θ | x, vessel) over K = 16 axial bins θ ∈ [0, π), supervised only inside
vessels with the background regularised towards a uniform, high-entropy
distribution so that orientation is *uninformative* rather than confidently wrong
inside the gaps the path search must cross. The segmenter itself is frozen and
unaware of the repair stage.

### 3.3 Path search: corridor-restricted lifted A\*
For each candidate a local corridor Ω_ij is built — the bounding box of the two
anchors dilated by m = 2·max(r_i, r_j, d_ij), clipped to the image, and
downsampled by an integer factor if any side exceeds 128 px. On the 8-connected
lifted grid (position × orientation bin) the transition cost of a move to
(x′, θ_k′) with step length Δs ∈ {1, √2} is

  w = [ −log Ṽ_I(x′) − λ_θ log q̃(θ_k′ | x′) ] · Δs + μ (Δθ)² / Δs,

with both probability maps clamped to [0, 1] + ε, ε = 10⁻⁴, so every cost is
finite. The last term is the discretisation of ∫ κ² ds; the wording used in the
manuscript is "Euler-elastica-inspired lifted shortest path", with no claim to
recover an exact continuous elastica. The heuristic h(x, θ) = w_min ‖x − x_target‖₂
with w_min the smallest per-unit-length cost achievable in the corridor is
consistent, hence admissible, so A\* returns the minimum-cost path **on the
discrete lifted graph** (optimality is claimed relative to that graph only).
Appearance evidence is taken from V_I, never from the segmenter's own
P_vessel — using −log P_vessel would penalise the true path most heavily exactly
inside the real gap and drive the search around it.

### 3.4 Scoring and calibration
A very small model (logistic regression or a two-layer MLP of width 32, < 5 K
parameters) maps a 13-component feature vector ψ(e) to a score: the optimal path
energy E\*(γ_ij) and its length-normalised form; the separation d_ij and
d_ij / r̄; the calibre mismatch |log(r_i/r_j)|; the tangent mismatch angle;
min_γ V_I and mean_γ V_I; min_γ q; local contrast; predicted local density; the
automatic disc-relative zone; and a flag for whether the far node is an
attachment port. Every feature is computable from (I, M̂, P) and the head
outputs; the feature builder takes no reference argument, so reference
information cannot enter by construction. The score is then **calibrated** into a
probability p_e by temperature scaling or isotonic regression fitted on the
validation split only — calibration is mandatory, because the acceptance rule
reads p_e as a probability rather than as a score.

### 3.5 Selection: prune, then match exactly
Two deterministic stages, in this order.

*Stage 1 — pre-matching pruning.* (i) Every edge with U(e) ≤ 0 is removed.
(ii) Corridor conflicts are resolved greedily: edges are visited in descending
U, and an edge is kept only if its corridor intersection-over-union with every
already-kept edge is at most θ_ov = 0.5. Two hard vetoes are applied alongside:
a path crossing a clearly unrelated vessel, and (in the typed extension) a
violated artery/vein constraint.

*Stage 2 — exact maximum-weight matching.* On the pruned edge set E_c′ over
V_c, maximise Σ_e U(e) x_e subject to Σ_{e∋v} x_e ≤ cap(v) = 1 for every node
and x_e ∈ {0, 1}. This is an ordinary maximum-weight matching and is solved
exactly by the blossom algorithm on a small edge set. Corridor exclusivity is
deliberately handled as a *pruning of the edge set beforehand* rather than as an
extra pairwise constraint, so that the problem actually solved is a standard
matching; the price is that stage 1 is greedy and the pruned set is not
guaranteed to be the U-optimal conflict-free subset.

Accepted paths are then painted into the mask with the local tube radius.

## 4. Why this is the right counterfactual for the audit

The audit needs an operator that changes **connectivity and as little else as
possible**, applied to the real predicted masks, so that the resulting biomarker
movement can be read as the topology-specific component of the measurement
error. This reconnector qualifies: it is additive only, capacity-limited,
geometry-constrained, and it never consults the reference. Its measured effect on
the primary panel is the "topology" axis of the decomposition. It is used here
purely as a measurement instrument — the audit makes no claim that this
reconnector is a good repair method, and the intervention experiments elsewhere
in the paper in fact find that none of the six candidate remedies, this one
included, gives a consistent fidelity improvement.

## 5. Provenance of the numbers quoted above

| Quantity | Where it is recorded |
|---|---|
| λ = 1.0, η = 0.1, τ = 0.5 (unused in this arm), μ = 2.0, λ_θ = 1.0, `risk_backend = "uniform"`, `orientation = "learned"` | `exp/runs/rigr/uniform/<ds>/seed<S>/summary.json` |
| candidate / positive-utility / accepted counts (HRF: 209.4 / 126.8 / 55.5 per image, 59.2 pruned by conflict, 0 vetoed) | same file, `mean` block |
| α = 6, τ₁ = 0.5, calibre gate log 2, θ_ov = 0.5, K = 16, ε = 10⁻⁴, 128-px corridor cap, < 5 K-parameter scorer, ≈ 0.3 M-parameter head | `paper/sections/03_method.tex` |
| per-image biomarker movement caused by the repair | `exp/results/pivot/e1_topology_image.csv` (`arm = rigr_uniform`) |

**No TODO and no companion-paper dependency remains in this description.**
