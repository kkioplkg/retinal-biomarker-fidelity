"""Synthetic ground-truth tests for the topology metrics and the TRR/FCR judge.

Every shape here has a topology that can be worked out by hand, so the tests
pin down the *conventions* (foreground 8-connectivity / background
4-connectivity, FOV restriction, tolerances) rather than just exercising the
code paths.

Run with::

    cd exp && python -m src.topo.tests_synthetic

Exit code 0 = all assertions passed.
"""

from __future__ import annotations

import sys
import traceback

import numpy as np
from scipy import ndimage as ndi

from src.topo import metrics as M
from src.topo import skeleton as S
from src.eval import trr_fcr as T

H = W = 128
_PASS: list = []
_FAIL: list = []


# --------------------------------------------------------------------------
# shape builders
# --------------------------------------------------------------------------
def _draw_lines(shape, segments, width=5):
    """Rasterise ``segments`` = [((r0,c0),(r1,c1)), ...] and dilate to ``width``."""
    from skimage.draw import line
    from skimage.morphology import disk

    m = np.zeros(shape, dtype=bool)
    for (r0, c0), (r1, c1) in segments:
        rr, cc = line(int(r0), int(c0), int(r1), int(c1))
        m[rr, cc] = True
    r = max(0, (int(width) - 1) // 2)
    if r:
        m = ndi.binary_dilation(m, structure=disk(r).astype(bool))
    return m


def y_vessel(shape=(H, W), width=5):
    """Y-shaped vessel: one stem, one bifurcation, three free ends."""
    return _draw_lines(
        shape,
        [((110, 64), (64, 64)), ((64, 64), (20, 30)), ((64, 64), (20, 98))],
        width=width,
    )


def ring(shape=(H, W), radius=34, thickness=5, centre=(64, 64)):
    """A closed annulus: one component with exactly one hole."""
    from skimage.draw import disk as _disk

    m = np.zeros(shape, dtype=bool)
    rr, cc = _disk(centre, radius, shape=shape)
    m[rr, cc] = True
    rr, cc = _disk(centre, radius - thickness, shape=shape)
    m[rr, cc] = False
    return m


def broken_line(shape=(H, W), row=64, gap=(58, 70), width=5):
    """A straight vessel cut in two by a gap of background."""
    m = _draw_lines(shape, [((row, 12), (row, shape[1] - 12))], width=width)
    m[:, gap[0] : gap[1]] = False
    return m


def straight_line(shape=(H, W), row=64, width=5):
    return _draw_lines(shape, [((row, 12), (row, shape[1] - 12))], width=width)


# --------------------------------------------------------------------------
# assertion helper
# --------------------------------------------------------------------------
def check(name, cond, detail=""):
    if cond:
        _PASS.append(name)
        print(f"  PASS  {name}" + (f"   [{detail}]" if detail else ""))
    else:
        _FAIL.append(name)
        print(f"  FAIL  {name}   [{detail}]")


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) <= tol


# --------------------------------------------------------------------------
# tests
# --------------------------------------------------------------------------
def test_betti():
    print("\n[1] Betti numbers (fg 8-connected / bg 4-connected, inside FOV)")
    y = y_vessel()
    check("Y vessel beta = (1, 0)", S.betti_numbers(y) == (1, 0), str(S.betti_numbers(y)))

    r = ring()
    check("ring beta = (1, 1)", S.betti_numbers(r) == (1, 1), str(S.betti_numbers(r)))

    two_rings = ring(centre=(40, 40), radius=24) | ring(centre=(95, 95), radius=24)
    check(
        "two rings beta = (2, 2)",
        S.betti_numbers(two_rings) == (2, 2),
        str(S.betti_numbers(two_rings)),
    )

    b = broken_line()
    check("broken line beta = (2, 0)", S.betti_numbers(b) == (2, 0), str(S.betti_numbers(b)))

    check("empty mask beta = (0, 0)", S.betti_numbers(np.zeros((H, W), bool)) == (0, 0))
    full = np.ones((H, W), bool)
    check("full mask beta = (1, 0)", S.betti_numbers(full) == (1, 0), str(S.betti_numbers(full)))

    # a hole that touches the FOV border is NOT a hole
    fov = np.zeros((H, W), bool)
    fov[20:108, 20:108] = True
    ring_cut = ring()
    ring_cut[64:, 64] = False  # break the annulus so its interior reaches out
    b0, b1 = S.betti_numbers(ring_cut, fov)
    check("cut ring inside FOV has no hole", (b0, b1) == (1, 0), f"({b0}, {b1})")

    # FOV clipping removes the part of the mask outside it
    fov_small = np.zeros((H, W), bool)
    fov_small[0:40, :] = True
    b0s, _ = S.betti_numbers(y, fov_small)
    check("FOV clipping splits the Y into 2 arms", b0s == 2, f"beta0={b0s}")


def test_skeleton():
    print("\n[2] Skeleton primitives")
    y = y_vessel()
    skel = S.skeletonize_mask(y)
    ends = S.endpoints(skel)
    cen, _, deg = S.junction_clusters(skel)
    check("Y skeleton has 3 endpoints", len(ends) == 3, f"n={len(ends)}")
    check("Y skeleton has 1 junction cluster", len(cen) == 1, f"n={len(cen)}")
    check("Y junction has degree 3", len(deg) == 1 and deg[0] == 3, f"deg={deg.tolist()}")
    check(
        "Y junction near (64, 64)",
        len(cen) == 1 and np.hypot(cen[0][0] - 64, cen[0][1] - 64) < 6,
        str(np.round(cen, 1).tolist()),
    )

    # spur pruning: a 2 px stub hanging off a straight skeleton is removed
    line_skel = np.zeros((41, 41), bool)
    line_skel[20, 5:36] = True
    spur = line_skel.copy()
    spur[19, 20] = True
    spur[18, 20] = True
    check("un-pruned spur creates a junction", len(S.junctions(spur)) > 0)
    pruned = S.prune_spurs(spur, min_branch_px=3)
    check("prune_spurs removes the 2 px spur", np.array_equal(pruned, line_skel))
    raw = S.skeletonize_mask(y, prune=False)
    check(
        "pruning keeps >=95% of a clean skeleton",
        S.prune_spurs(raw, 3).sum() > 0.95 * raw.sum(),
        f"{S.prune_spurs(raw, 3).sum()}/{raw.sum()}",
    )

    # radius / diameter
    rad = S.local_radius(straight_line(width=9))
    check("local_radius of a 9 px vessel ~ 4.5-5", 4.0 <= rad.max() <= 5.5, f"max={rad.max():.2f}")
    check("local_diameter = 2 * local_radius", approx(S.local_diameter(y).max(), 2 * S.local_radius(y).max(), 1e-6))

    lab, n = S.connected_components(broken_line())
    check("connected_components finds 2 fragments", n == 2, f"n={n}")


def test_summary():
    print("\n[3] skan graph + branch summary")
    y = y_vessel()
    skel = S.skeletonize_mask(y)
    try:
        df = S.summarize(skel, mask=y)
    except ImportError as exc:  # skan missing -> report but do not crash the file
        check("skan available", False, str(exc))
        return
    types = df["branch_type"].astype(int)
    check("summary has >= 3 branches", len(df) >= 3, f"n={len(df)}")
    check(
        "3 junction-to-endpoint branches (type 1)",
        int((types == 1).sum()) == 3,
        f"types={sorted(types.tolist())}",
    )
    check(
        "mean_radius ~ 2.5 for a 5 px vessel",
        bool((df["mean_radius"] > 1.5).all() and (df["mean_radius"] < 4.0).all()),
        str(np.round(df["mean_radius"].to_numpy(), 2).tolist()),
    )
    check("branch_distance > 0", bool((df["branch_distance"] > 10).all()))
    check("endpoint coords present", not df[["src_row", "src_col", "dst_row", "dst_col"]].isna().any().any())
    empty = S.summarize(np.zeros((16, 16), bool))
    check("empty skeleton -> empty summary", len(empty) == 0)


def test_pixel_metrics():
    print("\n[4] Pixel metrics")
    y = y_vessel()
    check("dice(m, m) == 1", approx(M.dice(y, y), 1.0))
    check("iou(m, m) == 1", approx(M.iou(y, y), 1.0))
    check("f1 == dice", approx(M.f1(y, y), M.dice(y, y)))
    empty = np.zeros((H, W), bool)
    check("dice(empty, empty) == 1", approx(M.dice(empty, empty), 1.0))
    check("dice(empty, gt) == 0", approx(M.dice(empty, y), 0.0))
    check("sen(empty, gt) == 0", approx(M.sen(empty, y), 0.0))
    check("spe(empty, gt) == 1", approx(M.spe(empty, y), 1.0))
    check("sen with empty GT is NaN", np.isnan(M.sen(y, empty)))

    prob = np.where(y, 0.9, 0.1)
    check("auc of a perfect prob map == 1", approx(M.auc(prob, y), 1.0))
    check("pr_auc of a perfect prob map == 1", approx(M.pr_auc(prob, y), 1.0))
    check("auc with single-class GT is NaN", np.isnan(M.auc(prob, empty)))
    check("auc with prob=None is NaN", np.isnan(M.auc(None, y)))

    # FOV really is honoured: errors outside the FOV must not count
    fov = np.zeros((H, W), bool)
    fov[:, :64] = True
    bad = y.copy()
    bad[:, 64:] = ~bad[:, 64:]
    check("errors outside the FOV are ignored", approx(M.dice(bad, y, fov), 1.0), f"{M.dice(bad, y, fov):.4f}")
    check("errors inside the FOV are counted", M.dice(bad, y) < 0.2, f"{M.dice(bad, y):.4f}")


def test_cldice():
    print("\n[5] clDice")
    y = y_vessel()
    check("clDice(m, m) == 1", approx(M.cldice(y, y), 1.0), f"{M.cldice(y, y):.6f}")
    check("clDice(ring, ring) == 1", approx(M.cldice(ring(), ring()), 1.0))
    check("clDice(empty, empty) == 1", approx(M.cldice(np.zeros((H, W), bool), np.zeros((H, W), bool)), 1.0))
    check("clDice(empty, gt) == 0", approx(M.cldice(np.zeros((H, W), bool), y), 0.0))
    b = broken_line()
    full = straight_line()
    v = M.cldice(b, full)
    check("clDice drops when a vessel is broken", 0.5 < v < 1.0, f"{v:.4f}")
    check("clDice is symmetric-ish and <= 1", M.cldice(full, b) <= 1.0)


def test_betti_error_and_components():
    print("\n[6] Betti error / component count error")
    full = straight_line()
    b = broken_line()
    be = M.betti_error(b, full)
    check("beta0 error of one break == 1", approx(be["beta0_err"], 1.0), str(be))
    check("beta1 error == 0", approx(be["beta1_err"], 0.0))
    r = ring()
    be2 = M.betti_error(straight_line(), r)
    check("missing a loop gives beta1 error 1", approx(be2["beta1_err"], 1.0), str(be2))
    check("component_count_error(break) == 1", approx(M.component_count_error(b, full), 1.0))
    check("component_count_error(m, m) == 0", approx(M.component_count_error(full, full), 0.0))


def test_junction_f1():
    print("\n[7] Junction F1")
    y = y_vessel()
    check("junction_f1(m, m) == 1", approx(M.junction_f1(y, y), 1.0))

    shifted = np.zeros_like(y)
    shifted[2:, 2:] = y[:-2, :-2]  # shift by (2, 2) -> ~2.83 px, inside tol 3
    v3 = M.junction_f1(shifted, y, tol_px=3.0)
    check("shifted junction matches within tol 3", approx(v3, 1.0), f"{v3:.3f}")
    v1 = M.junction_f1(shifted, y, tol_px=1.0)
    check("shifted junction misses at tol 1", approx(v1, 0.0), f"{v1:.3f}")

    line = straight_line()
    check("no junction on either side -> NaN", np.isnan(M.junction_f1(line, line)))
    check("junction in GT only -> 0", approx(M.junction_f1(line, y), 0.0))

    # a Y with a second bifurcation: half of the GT junctions found -> F1 = 2/3
    y2 = y_vessel() | _draw_lines((H, W), [((40, 46), (20, 20))], width=5)
    v = M.junction_f1(y, y2, tol_px=3.0)
    check("1 of 2 GT junctions found -> F1 = 2/3", approx(v, 2.0 / 3.0, 1e-6), f"{v:.4f}")


def test_bcs():
    print("\n[8] BCS (bifurcation connectedness, our re-implementation)")
    y = y_vessel()
    check("BCS(m, m) == 1", approx(M.bcs(y, y), 1.0), f"{M.bcs(y, y):.3f}")

    # sever one arm ~9 px from the junction (64, 64): the probe point, 15 steps
    # along the GT skeleton, then lands on the distal fragment -- a *different*
    # predicted connected component -- so the bifurcation is not connected.
    from skimage.draw import disk as _disk

    severed = y.copy()
    rr, cc = _disk((57, 58), 4, shape=(H, W))  # on the (64,64) -> (20,30) arm
    severed[rr, cc] = False
    b0_sev, _ = S.betti_numbers(severed)
    v = M.bcs(severed, y)
    check("severing an arm splits the mask", b0_sev == 2, f"beta0={b0_sev}")
    check("BCS == 0 after severing the only bifurcation", approx(v, 0.0), f"{v:.3f}")
    check("clDice stays high while BCS collapses", M.cldice(severed, y) > 0.8, f"clDice={M.cldice(severed, y):.3f}")

    # cutting the arm far from the junction (> probe_px) leaves BCS at 1
    far = y.copy()
    far[20:26, :] = False
    check("distal cut beyond the probe radius keeps BCS == 1", approx(M.bcs(far, y), 1.0), f"{M.bcs(far, y):.3f}")

    check("BCS with no GT bifurcation is NaN", np.isnan(M.bcs(straight_line(), straight_line())))
    check("BCS(empty pred) == 0", approx(M.bcs(np.zeros((H, W), bool), y), 0.0))


def test_evaluate_all():
    print("\n[9] evaluate_all")
    y = y_vessel()
    prob = np.where(y, 0.9, 0.1)
    res = M.evaluate_all(prob, y, y, None)
    check("evaluate_all returns floats", all(isinstance(v, float) for v in res.values()))
    check("perfect prediction: dice == 1", approx(res["dice"], 1.0))
    check("perfect prediction: cldice == 1", approx(res["cldice"], 1.0))
    check("perfect prediction: bcs == 1", approx(res["bcs"], 1.0))
    check("perfect prediction: betti errors == 0", approx(res["beta0_err"], 0) and approx(res["beta1_err"], 0))
    empty = np.zeros((H, W), bool)
    res0 = M.evaluate_all(np.zeros((H, W)), empty, y, None)
    check("empty prediction does not raise", isinstance(res0, dict))
    check("empty prediction: dice == 0", approx(res0["dice"], 0.0))
    check("empty prediction: bcs == 0", approx(res0["bcs"], 0.0))
    print("    perfect:", {k: round(v, 3) for k, v in list(res.items())[:8]})


def test_trr_fcr():
    print("\n[10] TRR / FCR")
    gt = straight_line()
    base = broken_line()  # the same vessel with a 12 px gap
    events = T.count_repairable_events(base, gt)
    check("one gap -> one repairable event", len(events) == 1, f"n={len(events)}")
    check("the event separates 2 predicted components", len(events[0]["components"]) == 2)

    good = T.CandidateEdge(
        path=T.rasterize_path([(64, 57), (64, 70)]), radius=2.0, edge_id="good", score=0.9
    )
    res = T.trr_fcr([good], base, gt, return_details=True)
    rec = res["records"][0]
    check("correct bridge: graph criterion", rec["crit_graph"], f"geo={rec['geodesic_len']:.1f}")
    check("correct bridge: corridor criterion", rec["crit_corridor"], f"overlap={rec['overlap']:.2f}")
    check("correct bridge: connectivity criterion", rec["crit_connect"], f"merged={rec['comp_merged']}")
    check("correct bridge is a TRUE repair", rec["is_true_repair"])
    check("TRR_recall == 1", approx(res["TRR_recall"], 1.0), str(res["TRR_recall"]))
    check("TRR_precision == 1", approx(res["TRR_precision"], 1.0))
    check("FCR == 0", approx(res["FCR"], 0.0))
    check("n_should == 1, n_accepted == 1", res["n_should"] == 1 and res["n_accepted"] == 1)

    # ---- false bridge between two genuinely distinct vessels ----------------
    gt2 = straight_line(row=40) | straight_line(row=88)
    base2 = gt2.copy()
    ev2 = T.count_repairable_events(base2, gt2)
    check("intact prediction -> no repairable event", len(ev2) == 0, f"n={len(ev2)}")
    bad = T.CandidateEdge(
        path=T.rasterize_path([(43, 64), (85, 64)]), radius=2.0, edge_id="bad", score=0.8
    )
    res2 = T.trr_fcr([bad], base2, gt2, return_details=True)
    r2 = res2["records"][0]
    check("false bridge fails the graph criterion", not r2["crit_graph"], f"geo={r2['geodesic_len']}")
    check("false bridge fails the corridor criterion", not r2["crit_corridor"], f"overlap={r2['overlap']:.2f}")
    check("false bridge is NOT a true repair", not r2["is_true_repair"])
    check("FCR == 1 for a single false bridge", approx(res2["FCR"], 1.0))
    check("TRR_precision == 0", approx(res2["TRR_precision"], 0.0))
    check("TRR_recall is NaN when N_should == 0", np.isnan(res2["TRR_recall"]))

    # ---- a shortcut that skips a bend: right components, wrong corridor -----
    bend = _draw_lines((H, W), [((30, 20), (30, 100)), ((30, 100), (100, 100))], width=5)
    base3 = bend.copy()
    base3[28:33, 60:72] = False  # break the horizontal arm
    short = T.CandidateEdge(
        path=T.rasterize_path([(30, 59), (100, 99)]), radius=2.0, edge_id="shortcut"
    )
    res3 = T.trr_fcr([short], base3, bend, return_details=True)
    r3 = res3["records"][0]
    check("shortcut fails the corridor criterion", not r3["crit_corridor"], f"overlap={r3['overlap']:.2f}")
    check("shortcut is not a true repair", not r3["is_true_repair"])

    # ---- mixed set: one true, one false ------------------------------------
    res4 = T.trr_fcr([good, T.CandidateEdge(path=T.rasterize_path([(20, 20), (20, 40)]), edge_id="nowhere")], base, gt)
    check("mixed set: n_true == 1 of 2", res4["n_true"] == 1, str(res4))
    check("mixed set: FCR == 0.5", approx(res4["FCR"], 0.5), str(res4["FCR"]))
    check("mixed set: TRR_recall == 1", approx(res4["TRR_recall"], 1.0))

    # ---- empty acceptance set ----------------------------------------------
    res5 = T.trr_fcr([], base, gt)
    check("empty A: FCR == 0, recall == 0", approx(res5["FCR"], 0.0) and approx(res5["TRR_recall"], 0.0), str(res5))
    check("empty A: n_should still counted", res5["n_should"] == 1)

    # ---- one-to-one matching: two edges closing the same gap ---------------
    good2 = T.CandidateEdge(
        path=T.rasterize_path([(65, 57), (65, 70)]), radius=2.0, edge_id="good2", score=0.7
    )
    res6 = T.trr_fcr([good, good2], base, gt)
    check("two edges, one event -> m == 1", res6["n_matched"] == 1, str(res6))
    check("recall stays <= 1", res6["TRR_recall"] <= 1.0)


def main():
    tests = [
        test_betti,
        test_skeleton,
        test_summary,
        test_pixel_metrics,
        test_cldice,
        test_betti_error_and_components,
        test_junction_f1,
        test_bcs,
        test_evaluate_all,
        test_trr_fcr,
    ]
    print("=" * 72)
    print("synthetic topology / metric tests")
    print("=" * 72)
    for t in tests:
        try:
            t()
        except Exception:
            _FAIL.append(t.__name__ + " (exception)")
            print(f"  ERROR in {t.__name__}:")
            traceback.print_exc()
    print("\n" + "=" * 72)
    print(f"{len(_PASS)} passed, {len(_FAIL)} failed")
    if _FAIL:
        for name in _FAIL:
            print("  FAILED:", name)
    print("=" * 72)
    return 1 if _FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
