#!/usr/bin/env bash
set -euo pipefail
while ps -p 3672237 > /dev/null; do
  sleep 60
done
cd /data/ganyw/vla_grasp_agent/data/kitti
unzip -n data_object_velodyne.zip
unzip -n data_object_label_2.zip
unzip -n data_object_calib.zip
if [ -f data_object_image_2.zip ]; then
  unzip -n data_object_image_2.zip
fi
cp -r /data/ganyw/vla_grasp_agent/OpenPCDet/data/kitti/ImageSets /data/ganyw/vla_grasp_agent/data/kitti/
echo "STEP2_DONE $(date -u +%Y-%m-%dT%H:%M:%SZ)"
