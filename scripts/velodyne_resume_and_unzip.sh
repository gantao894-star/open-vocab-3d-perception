#!/usr/bin/env bash
set -u

ZIP="/data/ganyw/vla_grasp_agent/data/kitti/data_object_velodyne.zip"
TOTAL=28750710812
URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip"
RESUME_LOG="/data/ganyw/vla_grasp_agent/logs/kitti_velodyne_resume.log"
STEP_LOG="/data/ganyw/vla_grasp_agent/logs/kitti_step2_finalize.log"
OUT_DIR="/data/ganyw/vla_grasp_agent/data/kitti"
DONE_FLAG="/data/ganyw/vla_grasp_agent/logs/.kitti_velodyne_unzip_done"

mkdir -p "$(dirname "$RESUME_LOG")"

echo "[$(date '+%F %T')] velodyne supervisor start" >> "$STEP_LOG"

while true; do
  cur=$(stat -c %s "$ZIP" 2>/dev/null || echo 0)
  rem=$((TOTAL - cur))
  if [ "$rem" -lt 0 ]; then
    rem=0
  fi
  echo "[$(date '+%F %T')] velodyne progress ${cur}/${TOTAL} rem=${rem}" >> "$STEP_LOG"

  if [ "$cur" -ge "$TOTAL" ]; then
    break
  fi

  (
    cd "$OUT_DIR" || exit 1
    curl -L --fail --retry 8 --retry-delay 5 -r "${cur}-" "$URL" >> data_object_velodyne.zip
  ) >> "$RESUME_LOG" 2>&1

  sleep 3
done

if unzip -tq "$ZIP" >/dev/null 2>&1; then
  unzip -o -q "$ZIP" -d "$OUT_DIR" >> "$STEP_LOG" 2>&1
  echo "[$(date '+%F %T')] velodyne unzip complete" >> "$STEP_LOG"
  touch "$DONE_FLAG"
else
  echo "[$(date '+%F %T')] velodyne zip test failed at final check" >> "$STEP_LOG"
  exit 1
fi
