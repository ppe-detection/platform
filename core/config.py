"""
Configuration management for Edge Controller Service.

Handles environment variables, thresholds, and service configuration.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


@dataclass
class CameraConfig:
    """Configuration for a single camera."""
    id: str
    source: str  # RTSP URL or USB device index
    enabled: bool = True


@dataclass
class Config:
    """Main configuration class."""
    
    # Supabase Configuration
    supabase_url: str = field(default_factory=lambda: os.getenv("SUPABASE_URL", ""))
    supabase_key: str = field(default_factory=lambda: os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", "")))
    supabase_service_role_key: Optional[str] = None
    
    # AI Detector Configuration
    detector_timeout: float = 5.0
    use_mock_detector: bool = False  # Flag to use mock detection
    
    # Camera Configuration
    cameras: List[CameraConfig] = None
    frame_width: int = 640
    frame_height: int = 480
    fps: int = 10  # Frames per second to process
    
    # Violation Detection Configuration
    violation_debounce_seconds: float = 2.0
    violation_cooldown_seconds: float = 5.0  # Time before same violation can trigger again
    
    # PPE Requirements (can be overridden by supervisor preferences from Supabase)
    require_goggles: bool = True
    require_lab_coat: bool = True
    require_gloves: bool = False
    
    # Person-PPE Association
    iou_threshold: float = 0.3  # Intersection over Union for matching PPE to people
    
    # Storage Configuration
    snapshot_storage_bucket: str = "violation-snapshots"
    snapshot_quality: int = 85  # JPEG quality (1-100)
    
    # Reconnection Settings
    camera_reconnect_delay: float = 5.0
    detector_reconnect_delay: float = 3.0
    max_reconnect_attempts: int = 10
    
    # Logging
    log_level: str = "INFO"
    
    def __post_init__(self):
        """Initialize configuration from environment variables."""
        # Supabase - handled by default_factory but validated here
        if not self.supabase_url:
            self.supabase_url = os.getenv("SUPABASE_URL", "")
        if not self.supabase_key:
            self.supabase_key = os.getenv("SUPABASE_KEY", os.getenv("SUPABASE_ANON_KEY", ""))

        self.supabase_service_role_key = os.getenv(
            "SUPABASE_SERVICE_ROLE_KEY",
            self.supabase_service_role_key
        )
        
        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "SUPABASE_URL and SUPABASE_KEY (or SUPABASE_ANON_KEY) must be set in environment"
            )
        
        # AI Detector
        self.detector_timeout = float(
            os.getenv("DETECTOR_TIMEOUT", str(self.detector_timeout))
        )
        self.use_mock_detector = os.getenv(
            "USE_MOCK_DETECTOR", str(self.use_mock_detector)
        ).lower() == "true"
        
        # Camera Configuration
        camera_sources = os.getenv("CAMERA_SOURCES", "")
        if camera_sources:
            # Format: "camera1:rtsp://...,camera2:0" (RTSP URL or USB index)
            self.cameras = []
            for cam_str in camera_sources.split(","):
                parts = cam_str.strip().split(":", 1)
                if len(parts) == 2:
                    cam_id, source = parts
                    self.cameras.append(CameraConfig(id=cam_id, source=source))
        else:
            # Default: single USB camera at index 0
            self.cameras = [CameraConfig(id="camera_0", source="0")]
        
        # Frame settings
        self.frame_width = int(os.getenv("FRAME_WIDTH", str(self.frame_width)))
        self.frame_height = int(
            os.getenv("FRAME_HEIGHT", str(self.frame_height))
        )
        self.fps = int(os.getenv("FPS", str(self.fps)))
        
        # Violation settings
        self.violation_debounce_seconds = float(
            os.getenv(
                "VIOLATION_DEBOUNCE_SECONDS",
                str(self.violation_debounce_seconds)
            )
        )
        self.violation_cooldown_seconds = float(
            os.getenv(
                "VIOLATION_COOLDOWN_SECONDS",
                str(self.violation_cooldown_seconds)
            )
        )
        
        # PPE Requirements
        self.require_goggles = os.getenv(
            "REQUIRE_GOGGLES", str(self.require_goggles)
        ).lower() == "true"
        self.require_lab_coat = os.getenv(
            "REQUIRE_LAB_COAT", str(self.require_lab_coat)
        ).lower() == "true"
        self.require_gloves = os.getenv(
            "REQUIRE_GLOVES", str(self.require_gloves)
        ).lower() == "true"
        
        # IOU threshold
        self.iou_threshold = float(
            os.getenv("IOU_THRESHOLD", str(self.iou_threshold))
        )
        
        # Storage
        self.snapshot_storage_bucket = os.getenv(
            "SNAPSHOT_STORAGE_BUCKET",
            self.snapshot_storage_bucket
        )
        self.snapshot_quality = int(
            os.getenv("SNAPSHOT_QUALITY", str(self.snapshot_quality))
        )
        
        # Reconnection
        self.camera_reconnect_delay = float(
            os.getenv(
                "CAMERA_RECONNECT_DELAY",
                str(self.camera_reconnect_delay)
            )
        )
        self.detector_reconnect_delay = float(
            os.getenv(
                "DETECTOR_RECONNECT_DELAY",
                str(self.detector_reconnect_delay)
            )
        )
        self.max_reconnect_attempts = int(
            os.getenv(
                "MAX_RECONNECT_ATTEMPTS",
                str(self.max_reconnect_attempts)
            )
        )
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
