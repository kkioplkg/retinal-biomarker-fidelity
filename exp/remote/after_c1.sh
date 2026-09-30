#!/bin/bash
# wait for remote C1 to finish, then run stats_c1 + btr remotely and pull results back
R="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=30 user@GPU_HOST"
for i in $(seq 1 400); do
  n=$($R 'pgrep -fc "src.c1.run_c1"' 2>/dev/null | tr -d '\r')
  if [ "${n:-1}" = "0" ]; then echo "C1 finished after $i polls"; break; fi
  sleep 60
done
$R 'cd /path/to/workdir/medical1/exp && export PATH=/path/to/workdir/c1venv/bin:$PATH && tail -15 ../logs/c1_full.log && ls -la results/c1_events.parquet results/c1_images.csv 2>&1 | tail -2 && nohup bash -c "python -m src.c1.stats_c1 --events results/c1_events.parquet --images results/c1_images.csv > ../logs/c1_stats.log 2>&1; python -m src.c1.btr --events results/c1_events_harm.parquet > ../logs/c1_btr.log 2>&1; echo DONE >> ../logs/c1_btr.log" > /dev/null 2>&1 &'
for i in $(seq 1 240); do
  if $R 'grep -q DONE /path/to/workdir/medical1/logs/c1_btr.log' 2>/dev/null; then echo "stats+btr done after $i polls"; break; fi
  sleep 60
done
mkdir -p results_remote figs_remote
$R 'cd /path/to/workdir/medical1/exp && tar czf - results/c1_*.csv results/c1_*.parquet results/c1_*.json results/tab1_* results/fig2_* results/c1_mixedlm_* figs runs/btr ../logs/c1_stats.log ../logs/c1_btr.log 2>/dev/null' | tar xzf - -C results_remote
ls -R results_remote | head -40; tail -20 results_remote/logs/c1_btr.log 2>/dev/null; tail -5 results_remote/logs/c1_stats.log 2>/dev/null
