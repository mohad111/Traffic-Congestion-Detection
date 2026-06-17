# Traffic Congestion Detection using YOLO11 and ByteTrack

## Overview

This project performs real-time traffic congestion detection using **YOLO11** for vehicle detection and **ByteTrack** for multi-object tracking.

The system detects vehicles in a video stream, tracks them across frames, estimates traffic density and vehicle movement, and determines whether traffic conditions are **FLOWING** or **CONGESTED**.


---

## Features

* Real-time vehicle detection using YOLO11
* Multi-object tracking with ByteTrack
* Vehicle counting inside a Region of Interest (ROI)
* Traffic congestion scoring

---


## Installation


```bash
pip install git+https://github.com/ultralytics/ultralytics.git@main
```


---

## Running the Project


```bash
python traffic_congestion.py \
    --source traffic.mp4 \
    --output output_video.mp4 \
    --model yolo11s.pt
```
---

## Example Output

Sample output video:

https://github.com/<username>/Traffic-Congestion-Detection/assets/OutPut_video.mp4


