#!/bin/sh
# Does bulk offline work hurt an interactive upload? Enqueues a temporal backfill of the whole
# library (offline, priority < 0), uploads one video, and measures its time-to-first-searchable
# and time-to-ready under three worker layouts:
#   shared   : 2 general workers (offline and interactive compete)
#   budget   : 2 general workers, OFFLINE_MAX_RUNNING=1 (at most one offline job at a time)
#   reserved : 1 general worker + 1 worker with --min-priority 0 (never takes offline work)
# usage: scripts/bench_isolation.sh <shared|budget|reserved> [file] [--no-load]
MODE="${1:-shared}"; FILE="${2:-eval/videos/db_lecture.mp4}"; NOLOAD="$3"
cd "$(dirname "$0")/.."
export PYTHONUNBUFFERED=1
pkill -9 -f "scenepeek worker" 2>/dev/null; sleep 1

if [ "$NOLOAD" != "--no-load" ]; then
  (cd backend && uv run scenepeek backfill temporal --force | tail -1)
fi

case "$MODE" in
  shared)   (cd backend && uv run scenepeek worker > /tmp/iso_w1.log 2>&1 &)
            (cd backend && uv run scenepeek worker > /tmp/iso_w2.log 2>&1 &) ;;
  budget)   (cd backend && OFFLINE_MAX_RUNNING=1 uv run scenepeek worker > /tmp/iso_w1.log 2>&1 &)
            (cd backend && OFFLINE_MAX_RUNNING=1 uv run scenepeek worker > /tmp/iso_w2.log 2>&1 &) ;;
  reserved) (cd backend && uv run scenepeek worker > /tmp/iso_w1.log 2>&1 &)
            (cd backend && uv run scenepeek worker --min-priority 0 > /tmp/iso_w2.log 2>&1 &) ;;
  *) echo "unknown mode $MODE"; exit 1 ;;
esac
sleep 20  # let the offline work saturate the workers before the interactive upload lands

ID=$(scripts/upload.sh "$FILE" "iso-$MODE-$(date +%s)" | tail -1)
START=$(date +%s)
while :; do
  sleep 2
  S=$(curl -s localhost:8000/api/videos/$ID | python3 -c "import sys,json;print(json.load(sys.stdin)['status'])")
  [ "$S" = "ready" ] || [ "$S" = "failed" ] && break
  [ $(( $(date +%s) - START )) -gt 1800 ] && echo "timeout" && break
done
pkill -9 -f "scenepeek worker"
curl -s localhost:8000/api/videos/$ID | python3 - "$MODE" <<'PY'
import json, sys, datetime as dt
v = json.load(sys.stdin); mode = sys.argv[1]
p = lambda s: dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
ttfs = (p(v["first_searchable_at"]) - p(v["upload_completed_at"])).total_seconds()
ttr = (p(v["completed_at"]) - p(v["upload_completed_at"])).total_seconds() if v.get("completed_at") else None
out = {"mode": mode, "video_seconds": v["duration_s"], "chunks": v["chunk_count"],
       "time_to_first_searchable_s": round(ttfs, 1), "time_to_ready_s": round(ttr, 1) if ttr else None}
print(out)
import os; os.makedirs("eval/reports", exist_ok=True)
json.dump(out, open(f"eval/reports/bench_isolation_{mode}.json", "w"), indent=2)
PY
curl -s -X DELETE localhost:8000/api/videos/$ID -o /dev/null
