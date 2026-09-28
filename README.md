# CodeAlpha Object Detection and Tracking

## Project Overview

This project is an AI-based Object Detection and Tracking system developed as part of the CodeAlpha Internship.

The system uses YOLO for object detection and Deep SORT for tracking objects across video frames. It can work with both a live webcam and a prerecorded video.

## Features

- Real-time object detection
- Object tracking using Deep SORT
- Unique tracking IDs for detected objects
- Bounding boxes around detected objects
- Object class labels
- Webcam input
- Prerecorded video input
- Multiple objects can be detected and tracked simultaneously

## Technologies Used

- Python
- OpenCV
- YOLO
- Deep SORT
- NumPy

## Project Structure

```text
CodeAlpha_Object_Detection_Tracking
│
├── app.py
├── requirements.txt
├── README.md
├── yolo11n.pt
│
├── models
│
└── videos
    └── test_video.mp4