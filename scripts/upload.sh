#!/bin/sh
# usage: scripts_upload.sh <file> [title]
F="$1"; TITLE="${2:-$(basename "$F")}"
R=$(curl -s -X POST localhost:8000/api/videos -H 'content-type: application/json' -d "{\"filename\":\"$(basename "$F")\",\"content_type\":\"video/mp4\",\"title\":\"$TITLE\"}")
URL=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['upload_url'])")
ID=$(echo "$R" | python3 -c "import sys,json;print(json.load(sys.stdin)['video_id'])")
curl -s -o /dev/null -w "PUT %{http_code}\n" -X PUT -H 'content-type: video/mp4' --data-binary @"$F" "$URL"
curl -s -X POST localhost:8000/api/videos/$ID/complete >/dev/null && echo $ID
