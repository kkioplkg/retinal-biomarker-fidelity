"""RiGR -- Risk-guided Graph Repair (proposal v3 section 3.2, plan stage S4).

A post-hoc, backbone-agnostic repair stage whose only inputs are
``(image I, probability map P, predicted mask M_hat)``.  Nothing in this package
reads the reference annotation at inference time (feature contract F).

Modules
-------
``head``        micro multi-task head: vessel evidence ``V_I`` + conditional
                axial orientation ``q(theta | x, vessel)``; ``frangi_evidence``
                is the analytic ablation alternative.
``train_head``  CLI to train the head.
``candidates``  dangling endpoints, endpoint-endpoint and endpoint-port (T-type)
                candidates, attachment-port pseudo-nodes, corridor boxes.
``astar``       corridor-restricted lifted-state A* over ``(x, y, theta_k)``.
``synth_cuts``  uniform + failure-conditioned synthetic severances for training
                the pair scorer.
``scorer``      pair scorer (logistic regression / small MLP) + calibration.
``utility``     asymmetric expected utility, max-weight matching, hard vetoes,
                tube rasterisation.
``run_rigr``    end-to-end CLI.
"""

__all__ = [
    "head",
    "candidates",
    "astar",
    "synth_cuts",
    "scorer",
    "utility",
]
