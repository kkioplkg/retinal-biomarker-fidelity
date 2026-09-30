#!/bin/bash
# poll remote venv readiness, then launch the full C1 run with nohup
R="ssh -p SSH_PORT -o BatchMode=yes -o ServerAliveInterval=30 -o ConnectTimeout=30 user@GPU_HOST"
for i in $(seq 1 60); do
  if $R '/path/to/workdir/c1venv/bin/python -c "import skan,PVBM,lightgbm,statsmodels,pyarrow,cv2"' 2>/dev/null; then
    echo "venv ready after $i polls"; break
  fi
  sleep 60
done
$R 'cd /path/to/workdir/medical1/exp && mkdir -p ../logs && export PATH=/path/to/workdir/c1venv/bin:$PATH && nohup env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 python -m src.c1.run_c1 --datasets DRIVE CHASE_DB1 HRF FIVES --workers 28 --out_dir results > ../logs/c1_full.log 2>&1 &
sleep 90; tail -5 ../logs/c1_full.log; pgrep -fc "src.c1.run_c1"'
