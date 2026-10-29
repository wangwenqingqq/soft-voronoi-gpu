#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
mode=${1:-dev}
out=${2:-runs/raw/${mode}-$(date -u +%Y%m%dT%H%M%SZ)}
case "$mode" in dev|test) ;; *) echo 'Usage: run_campaign.sh dev|test [new-output-directory]' >&2; exit 2;; esac
: "${CUDA_VISIBLE_DEVICES:?Set CUDA_VISIBLE_DEVICES to one approved idle GPU index or UUID}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then echo 'Only one GPU is permitted' >&2; exit 2; fi
if [[ -e "$out" ]]; then echo "Refusing to overwrite $out" >&2; exit 2; fi
mkdir -p "$out"
exec 9>"/tmp/soft-voronoi-gpu-${UID}-${CUDA_VISIBLE_DEVICES}.lock"
flock -n 9 || { echo 'Campaign GPU lock is held' >&2; exit 3; }
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
python3 - "$out" "$mode" <<'PY'
import hashlib, json, pathlib, sys, subprocess
out,mode=pathlib.Path(sys.argv[1]),sys.argv[2]
files=['soft_voronoi_bench.cu','build/soft_voronoi_bench','scripts/prepare_data.py','scripts/summarize.py','scripts/run_campaign.sh','runs/CONTRACT.md','data/manifest.json','build/compile.command','build/compile.stderr']
for work in ('words','dna_clustered','dna_random'):
    files.extend([f'data/{work}/db.txt',f'data/{work}/{mode}.txt'])
manifest={'mode':mode, 'files':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in files},'clock_policy':'unchanged','timing':'resident single-query, query H2D to result D2H, synchronized'}
(out/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
PY
nvcc --version > "$out/toolkit.txt"
for work in words dna_clustered dna_random; do
  pids=$(nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-compute-apps=pid --format=csv,noheader,nounits)
  if [[ -n "$pids" ]]; then echo "Target GPU has active compute PIDs: $pids; stop, do not kill them" >&2; exit 4; fi
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,driver_version,compute_cap,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,power.limit,temperature.gpu --format=csv > "$out/$work-gpu-before.csv"
  printf '%s %s %s\n' "$(date -u +%FT%TZ)" "$mode" "$work" | tee -a "$out/progress.log"
  timeout 1200 taskset -c "${CPU_CORES:-0-7}" ./build/soft_voronoi_bench \
    --db "data/$work/db.txt" --queries "data/$work/$mode.txt" \
    --workload "$work" --sample-check 64 --n 100000 --m 512 --kb 4,8,16 --candidate-frac 0.02,0.05 \
    --repeats 3 --warmup 3 --output "$out/$work.jsonl" \
    > "$out/$work-stdout.log" 2> "$out/$work-stderr.log"
  nvidia-smi -i "$CUDA_VISIBLE_DEVICES" --query-gpu=name,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,temperature.gpu --format=csv > "$out/$work-gpu-after.csv"
done
printf '%s complete\n' "$(date -u +%FT%TZ)" | tee -a "$out/progress.log"
