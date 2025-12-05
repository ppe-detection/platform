"""
Camera Manager Service

Handles robust connection to multiple cameras (RTSP or USB),
frame capture, and reconnection logic.
"""

import asyncio
import logging
from typing import Dict, Optional
import cv2
import numpy as np

from core import Config, CameraConfig
from .ai_client import AIClient
from .violation_engine import ViolationEngine

logger = logging.getLogger(__name__)


class CameraStream:
    """Manages a single camera stream."""
    
    def __init__(
        self,
        config: CameraConfig,
        global_config: Config,
        ai_client: AIClient,
        violation_engine: ViolationEngine
    ):
        self.config = config
        self.global_config = global_config
        self.ai_client = ai_client
        self.violation_engine = violation_engine
        
        self.cap: Optional[cv2.VideoCapture] = None
        self.running = False
        self.frame_count = 0
        self.reconnect_attempts = 0
        
    def _parse_source(self) -> tuple:
        """Parse camera source to determine type and value."""
        source = self.config.source.strip()
        
        # Check if it's an RTSP URL
        if source.startswith(("rtsp://", "http://", "https://")):
            return "rtsp", source
        
        # Check if it's a local image file (for testing)
        if source.endswith((".jpg", ".jpeg", ".png")):
            return "image", source

        # Check if it's a USB device index (numeric)
        try:
            index = int(source)
            return "usb", index
        except ValueError:
            # Assume it's a file path or other OpenCV source
            return "file", source
    
    async def _connect(self) -> bool:
        """Connect to the camera."""
        source_type, source_value = self._parse_source()
        
        try:
            if source_type == "rtsp":
                # RTSP streams need special handling
                self.cap = cv2.VideoCapture(
                    source_value,
                    cv2.CAP_FFMPEG
                )
                # Set buffer size to reduce latency
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif source_type == "image":
                # Static image mode for testing
                self.cap = None # No video capture object needed
                # Load image once to verify
                img = cv2.imread(source_value)
                if img is None:
                    logger.error(f"Camera {self.config.id}: Failed to load image {source_value}")
                    return False
                logger.info(f"Camera {self.config.id}: Using static image {source_value}")
                return True
            else:
                # USB or file
                self.cap = cv2.VideoCapture(source_value)
            
            if not self.cap.isOpened():
                logger.error(
                    f"Camera {self.config.id}: Failed to open source {source_value}"
                )
                return False
            
            # Set frame properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.global_config.frame_width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.global_config.frame_height)
            
            # Test read
            ret, frame = self.cap.read()
            if not ret:
                logger.error(
                    f"Camera {self.config.id}: Failed to read test frame"
                )
                self.cap.release()
                self.cap = None
                return False
            
            logger.info(
                f"Camera {self.config.id}: Connected successfully "
                f"(source: {source_value})"
            )
            self.reconnect_attempts = 0
            return True
            
        except Exception as e:
            logger.error(
                f"Camera {self.config.id}: Connection error: {e}",
                exc_info=True
            )
            if self.cap:
                self.cap.release()
                self.cap = None
            return False
    
    async def _process_frame(self, frame: np.ndarray):
        """Process a single frame through AI and violation detection."""
        try:
            # Resize frame if needed
            if (frame.shape[1] != self.global_config.frame_width or
                frame.shape[0] != self.global_config.frame_height):
                frame = cv2.resize(
                    frame,
                    (self.global_config.frame_width,
                     self.global_config.frame_height)
                )
            
            # Send to AI detector
            detections = await self.ai_client.detect(frame)
            
            if detections:
                # Process violations
                await self.violation_engine.process_detections(
                    camera_id=self.config.id,
                    frame=frame,
                    detections=detections
                )
            
            self.frame_count += 1
            
        except Exception as e:
            logger.error(
                f"Camera {self.config.id}: Frame processing error: {e}",
                exc_info=True
            )
    
    async def run(self):
        """Main camera loop."""
        self.running = True
        source_type, source_value = self._parse_source()
        
        while self.running:
            if source_type == "image":
                # Logic for static image: read the file repeatedly
                frame = cv2.imread(source_value)
                if frame is None:
                    logger.error(f"Camera {self.config.id}: Failed to read image file")
                    await asyncio.sleep(1)
                    continue
                
                # Process frame
                await self._process_frame(frame)
                
                # Sleep to simulate FPS
                await asyncio.sleep(1.0 / self.global_config.fps)
                continue

            # Connect if not connected
            if self.cap is None or not self.cap.isOpened():
                if self.reconnect_attempts >= self.global_config.max_reconnect_attempts:
                    logger.error(
                        f"Camera {self.config.id}: Max reconnection attempts reached. "
                        "Stopping stream."
                    )
                    break
                
                logger.info(
                    f"Camera {self.config.id}: Attempting to connect "
                    f"(attempt {self.reconnect_attempts + 1})..."
                )
                
                connected = await self._connect()
                if not connected:
                    self.reconnect_attempts += 1
                    await asyncio.sleep(self.global_config.camera_reconnect_delay)
                    continue
            
            # Read frame
            ret, frame = self.cap.read()
            
            if not ret:
                # Handle End of Video File (Rewind for testing)
                if source_type == "file":
                    logger.info(f"Camera {self.config.id}: Video ended, rewinding to start...")
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                # Actual error for live streams
                logger.warning(
                    f"Camera {self.config.id}: Failed to read frame. "
                    "Attempting reconnection..."
                )
                if self.cap:
                    self.cap.release()
                    self.cap = None
                self.reconnect_attempts += 1
                await asyncio.sleep(self.global_config.camera_reconnect_delay)
                continue
            
            # Process frame (only at configured FPS)
            if self.frame_count % (30 // self.global_config.fps) == 0:
                await self._process_frame(frame)
            
            # Small delay to control frame rate
            await asyncio.sleep(1.0 / 30.0)  # Read at ~30fps, process at configured FPS
    
    async def stop(self):
        """Stop the camera stream."""
        self.running = False
        if self.cap:
            self.cap.release()
            self.cap = None
        logger.info(f"Camera {self.config.id}: Stream stopped")


class CameraManager:
    """Manages multiple camera streams."""
    
    def __init__(
        self,
        config: Config,
        ai_client: AIClient,
        violation_engine: ViolationEngine
    ):
        self.config = config
        self.ai_client = ai_client
        self.violation_engine = violation_engine
        
        self.streams: Dict[str, CameraStream] = {}
        self.tasks: Dict[str, asyncio.Task] = {}
    
    async def start_all_cameras(self):
        """Start all enabled cameras."""
        logger.info(f"Starting {len(self.config.cameras)} camera(s)...")
        
        for camera_config in self.config.cameras:
            if not camera_config.enabled:
                logger.info(f"Camera {camera_config.id}: Disabled, skipping")
                continue
            
            stream = CameraStream(
                camera_config,
                self.config,
                self.ai_client,
                self.violation_engine
            )
            self.streams[camera_config.id] = stream
            
            # Start camera task
            task = asyncio.create_task(stream.run())
            self.tasks[camera_config.id] = task
        
        logger.info(f"Started {len(self.tasks)} camera stream(s)")
    
    async def stop_all_cameras(self):
        """Stop all camera streams."""
        logger.info("Stopping all camera streams...")
        
        # Stop all streams
        for stream in self.streams.values():
            await stream.stop()
        
        # Cancel all tasks
        for task in self.tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.tasks:
            await asyncio.gather(*self.tasks.values(), return_exceptions=True)
        
        self.streams.clear()
        self.tasks.clear()
        logger.info("All camera streams stopped")
