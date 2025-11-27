"""
AI Client Service

Communicates with the local Dockerized Object Detector service.
"""

import asyncio
import logging
from typing import List, Dict, Optional
import cv2
import numpy as np
import aiohttp
import base64
from io import BytesIO
import random
from core import Config

logger = logging.getLogger(__name__)


class AIClient:
    """Client for communicating with the object detection model."""
    
    def __init__(self, config: Config):
        self.config = config
        self.detector_url = f"{config.detector_url}{config.detector_endpoint}"
        self.timeout = aiohttp.ClientTimeout(total=config.detector_timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        self.reconnect_attempts = 0
    
    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create HTTP session."""
        if self.session is None or self.session.closed:
            self.session = aiohttp.ClientSession(timeout=self.timeout)
        return self.session
    
    def _encode_frame(self, frame: np.ndarray) -> str:
        """Encode frame as base64 JPEG."""
        # Encode frame as JPEG
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 90]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            raise ValueError("Failed to encode frame as JPEG")
        
        # Convert to base64
        image_bytes = base64.b64encode(buffer).decode('utf-8')
        return image_bytes
    
    async def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Send frame to detector and get detections.
        
        Args:
            frame: OpenCV frame (numpy array)
            
        Returns:
            List of detections in format:
            [
                {
                    "class": "person",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95
                },
                ...
            ]
        """
        # Handle Mock Mode
        if self.config.use_mock_detector:
            # Simulate network delay
            # await asyncio.sleep(0.1)
            
            # Return a mock detection of a person
            # This ensures the ViolationEngine sees a person
            # Since no PPE is returned, it will trigger "missing PPE" violations
            h, w = frame.shape[:2]
            
            # Center box
            x1 = int(w * 0.3)
            y1 = int(h * 0.2)
            x2 = int(w * 0.7)
            y2 = int(h * 0.8)
            
            return [
                {
                    "class": "person",
                    "bbox": [x1, y1, x2, y2],
                    "confidence": 0.95
                }
            ]

        try:
            # Encode frame
            image_b64 = self._encode_frame(frame)
            
            # Prepare request
            payload = {
                "image": image_b64
            }
            
            # Send request
            session = await self._get_session()
            async with session.post(self.detector_url, json=payload) as response:
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(
                        f"Detector returned status {response.status}: {error_text}"
                    )
                    return []
                
                result = await response.json()
                
                # Parse detections
                detections = result.get("detections", [])
                
                # Reset reconnect attempts on success
                self.reconnect_attempts = 0
                
                return detections
                
        except asyncio.TimeoutError:
            logger.warning("Detector request timed out")
            self.reconnect_attempts += 1
            return []
        except aiohttp.ClientError as e:
            logger.error(f"Detector connection error: {e}")
            self.reconnect_attempts += 1
            
            # Close session to force reconnect
            if self.session:
                await self.session.close()
                self.session = None
            
            return []
        except Exception as e:
            logger.error(f"Detector error: {e}", exc_info=True)
            return []
    
    async def health_check(self) -> bool:
        """Check if detector service is healthy."""
        if self.config.use_mock_detector:
            return True
            
        try:
            health_url = f"{self.config.detector_url}/health"
            session = await self._get_session()
            
            async with session.get(health_url, timeout=aiohttp.ClientTimeout(total=2.0)) as response:
                return response.status == 200
        except Exception:
            return False
    
    async def close(self):
        """Close HTTP session."""
        if self.session and not self.session.closed:
            await self.session.close()
            self.session = None
