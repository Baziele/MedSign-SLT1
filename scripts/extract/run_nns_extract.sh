#!/bin/bash
# Keep NNS keypoint extraction running until 697/697 npz exist.
# Exits 0 only when complete (launchd relaunches on any other exit).
cd /Users/mazen/dev/projects/medsign || exit 1
KPD=data/jw/nwt-nns/keypoints
LOGF=logs/nns_extract.log
while :; do
  COUNT=$(ls $KPD/*.npz 2>/dev/null | wc -l)
  if [ "$COUNT" -ge 697 ]; then
    echo "$(date -u '+%F %T') complete: $COUNT/697" >> $LOGF
    exit 0
  fi
  # Reap leftovers of previous runs: SIGKILLed parents orphan workers that
  # wedge at 0% CPU (or finish videos unlogged) and steal CPUs from the
  # new run. Nothing legitimate runs jwextract except this wrapper.
  # NOTE: patterns must stay narrow -- a broad "jwextract|spawn_main"
  # match would kill admin shells that merely mention those words.
  pkill -f "jwextract\.py --workers" 2>/dev/null
  sleep 2
  pkill -9 -f "spawn_main; spawn_main" 2>/dev/null
  pkill -9 -f "jwextract\.py --workers" 2>/dev/null
  sleep 2
  caffeinate -i -s .venv312/bin/python -u scripts/extract/jwextract.py --workers 8 \
    --src /Users/mazen/dev/projects/medsign-data-pipeline/nns \
    --dst /Users/mazen/dev/projects/medsign/data/jw/nwt-nns >> logs/nns_extract.log 2>&1
  echo "$(date -u '+%F %T') exited, waiting 60s before retry" >> logs/nns_extract.log
  sleep 60
done
