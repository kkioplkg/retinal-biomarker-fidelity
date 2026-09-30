"""C1: vascular structural perturbation engine, matched controls and BTR.

Modules
-------
perturb   Algorithm 1 of proposal v3 section 3.1.3 (sever / bridge / truncate /
          caliber + pixel- and context-matched topology-neutral controls) and
          the Contract-F feature extractor ``phi``.
run_c1    CLI that runs the engine over the four datasets and writes
          ``exp/results/c1_events.parquet``.
stats_c1  Robust scales, H_net / H_dep, nested mixed-effects models, Fig. 2.
btr       Multi-output LightGBM risk heads (R_miss / R_false) and Exp1.
"""
