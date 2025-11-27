# Edge Controller Service

Headless Python service for ingesting video streams from cameras, processing frames with a local AI detector, and pushing PPE violation alerts to Supabase.

## Overview

This service runs on a dedicated edge device in the laboratory. It:
- Connects to local USB/IP cameras (RTSP or USB)
- Sends frames to a local Dockerized Object Detector (running at `http://localhost:8000/predict`)
- Processes detections to identify safety violations
- Pushes alerts and session data to Supabase in real-time
- Listens for "Start/Stop Session" commands from Supabase

## Requirements

- Python 3.11+
- Access to cameras (USB or RTSP)
- Local Docker container running the object detection model
- Supabase project with configured storage bucket and database tables

## Installation

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Copy `.env.example` to `.env` and configure:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Ensure the object detection model is running:
```bash
# The model should be accessible at http://localhost:8000/predict
```

## Configuration

### Environment Variables

See `.env.example` for all available configuration options.

Key settings:
- `SUPABASE_URL` and `SUPABASE_ANON_KEY`: Supabase project credentials
- `CAMERA_SOURCES`: Comma-separated list of cameras (format: `camera_id:source`)
- `DETECTOR_URL`: URL of the local AI detector service
- `FPS`: Frames per second to process (default: 10)
- `VIOLATION_DEBOUNCE_SECONDS`: Time violation must persist before alert (default: 2.0)

### Camera Sources

Cameras can be specified as:
- USB device index: `camera_0:0` (uses `/dev/video0` on Linux)
- RTSP URL: `camera_1:rtsp://192.168.1.100:554/stream`
- File path: `camera_2:/path/to/video.mp4`

Example:
```
CAMERA_SOURCES=camera_0:0,camera_1:rtsp://192.168.1.100:554/stream
```

## Usage

Run the service:
```bash
python main.py
```

The service will:
1. Connect to all configured cameras
2. Start processing frames through the AI detector
3. Listen for session start/stop commands from Supabase
4. Upload violation snapshots and create alert records

## Architecture

```
main.py
├── core/
│   └── config.py          # Configuration management
└── services/
    ├── camera_manager.py   # Camera connection and frame capture
    ├── ai_client.py        # Communication with AI detector
    ├── violation_engine.py # PPE violation detection logic
    └── cloud_sync.py       # Supabase interactions
```

## Features

### Robust Camera Management
- Automatic reconnection on camera failure
- Support for multiple cameras (USB and RTSP)
- Configurable frame rate processing

### Violation Detection
- Person-PPE association using IoU (Intersection over Union)
- Debouncing to prevent noise (violations must persist for 2+ seconds)
- Cooldown period to prevent duplicate alerts
- Configurable PPE requirements

### Cloud Sync
- Uploads violation snapshots to Supabase Storage
- Creates alert records in database
- Listens for session commands via Supabase Realtime

## Development

The service is designed to be headless and run as a systemd service or in a container. For development, run directly with Python.

## Troubleshooting

### Camera Connection Issues
- Check camera permissions (Linux: add user to `video` group)
- Verify camera source format in `.env`
- Check camera is not being used by another process

### AI Detector Connection Issues
- Verify detector is running: `curl http://localhost:8000/health`
- Check `DETECTOR_URL` and `DETECTOR_ENDPOINT` in `.env`
- Increase `DETECTOR_TIMEOUT` if detector is slow

### Supabase Connection Issues
- Verify `SUPABASE_URL` and `SUPABASE_ANON_KEY` are correct
- Check network connectivity
- Ensure storage bucket exists and is accessible


