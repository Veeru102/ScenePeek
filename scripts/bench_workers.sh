#!/bin/sh
# Measure indexing wall time for N workers: uploads the eval videos, starts N workers, waits for ready.
# usage: scripts/bench_workers.sh <N> [files...]
N="${1:-1}"; shift
FILES="${@:-eval/videos/db_lecture.mp4 eval/videos/dist_lecture.mp4}"
cd "$(dirname "$0")/.."
IDS=""
for f in $FILES; do
  IDS="$IDS $(scripts/upload.sh "$f" "bench-$N-$(basename "$f" .mp4)" | tail -1)"
done
START=$(date +%s)
for i in $(seq 1 "$N"); do (cd backend && PYTHONUNBUFFERED=1 uv run scenepeek worker > /tmp/bench_worker_$i.log 2>&1 &); done
while :; do
  sleep 2
  DONE=1
  for id in $IDS; do
    S=$(curl -s localhost:8000/api/videos/$id | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
    [ "$S" = "ready" ] || DONE=0
  done
  [ $DONE = 1 ] && break
  [ $(( $(date +%s) - START )) -gt 900 ] && echo "timeout" && break
done
END=$(date +%s)
pkill -9 -f "scenepeek worker"
TOTAL=$(for id in $IDS; do curl -s localhost:8000/api/videos/$id | python3 -c "import sys,json;print(json.load(sys.stdin)['duration_s'])"; done | paste -sd+ - | bc)
NV=$(echo $IDS | wc -w | tr -d ' ')
python3 -c "print(f'workers=$N videos=$NV video_seconds=$TOTAL wall_seconds=$((END-START)) realtime_factor={$TOTAL/($END-START):.2f}')"
mkdir -p eval/reports
python3 - <<PY
import json, platform, os
out = {"workers": $N, "videos": $NV, "video_seconds": $TOTAL, "wall_seconds": $((END-START)),
       "realtime_factor": round($TOTAL/($((END-START)) or 1), 2), "cpus": os.cpu_count(),
       "platform": platform.platform(), "load_avg_end": os.getloadavg()}
json.dump(out, open("eval/reports/bench_workers_$N.json", "w"), indent=2)
print("wrote eval/reports/bench_workers_$N.json")
PY
for id in $IDS; do curl -s -X DELETE localhost:8000/api/videos/$id -o /dev/null; done
