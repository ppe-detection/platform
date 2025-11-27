"""
Cloud Sync Service

Handles all Supabase interactions:
- Uploading violation snapshots to Storage
- Inserting alert records
- Listening for session start/stop commands
"""

import asyncio
import logging
from typing import Optional, Callable, Dict
from datetime import datetime
import cv2
import numpy as np
from supabase import create_client, Client, ClientOptions

from core import Config

logger = logging.getLogger(__name__)


class CloudSync:
    """Service for syncing data with Supabase."""
    
    def __init__(self, config: Config):
        self.config = config
        self.supabase: Optional[Client] = None
        self.session_listener_task: Optional[asyncio.Task] = None
        self.session_command_callback: Optional[Callable] = None
        self.current_session_id: Optional[str] = None
    
    async def initialize(self):
        """Initialize Supabase client."""
        try:
            # For realtime support in async contexts, sometimes we need special handling
            # But standard client usually works if we don't block the loop.
            # However, the error suggests 'sync' client issues.
            # We will try to use the client but acknowledge realtime might need
            # a different approach or just simple polling if realtime fails.
            
            self.supabase = create_client(
                self.config.supabase_url,
                self.config.supabase_key
            )
            
            logger.info("Supabase client initialized")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}", exc_info=True)
            raise
    
    def _encode_frame(self, frame: np.ndarray) -> bytes:
        """Encode frame as JPEG bytes."""
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.config.snapshot_quality]
        success, buffer = cv2.imencode('.jpg', frame, encode_param)
        
        if not success:
            raise ValueError("Failed to encode frame as JPEG")
        
        return buffer.tobytes()
    
    async def upload_violation(
        self,
        session_id: str,
        camera_id: str,
        person_id: str,
        missing_ppe: str,
        frame: np.ndarray,
        bbox: list
    ):
        """
        Upload violation snapshot and create alert record.
        
        Args:
            session_id: Active lab session ID
            camera_id: Camera that detected the violation
            person_id: ID of the person with violation
            missing_ppe: Type of missing PPE (e.g., "goggles")
            frame: Frame snapshot (numpy array)
            bbox: Bounding box [x1, y1, x2, y2]
        """
        try:
            # Run blocking Supabase calls in a thread to avoid blocking asyncio loop
            await asyncio.to_thread(
                self._upload_violation_sync,
                session_id,
                camera_id,
                person_id,
                missing_ppe,
                frame,
                bbox
            )
            
        except Exception as e:
            logger.error(f"Failed to upload violation: {e}", exc_info=True)

    def _upload_violation_sync(
        self,
        session_id: str,
        camera_id: str,
        person_id: str,
        missing_ppe: str,
        frame: np.ndarray,
        bbox: list
    ):
        """Synchronous implementation of upload."""
        # Encode frame
        image_bytes = self._encode_frame(frame)
        
        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{session_id}/{camera_id}/{timestamp}_{person_id}_{missing_ppe}.jpg"
        
        # Upload to Supabase Storage
        response = self.supabase.storage.from_(
            self.config.snapshot_storage_bucket
        ).upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/jpeg"}
        )
        
        logger.info(f"Uploaded violation snapshot: {filename}")
        
        # Create alert record in database
        alert_data = {
            "session_id": session_id,
            "violation_type": missing_ppe,
            "image_path": filename,
            "created_at": datetime.now().isoformat(),
        }
        
        # Insert into alerts table
        result = self.supabase.table("alerts").insert(alert_data).execute()
        
        logger.info(f"Created violation alert: {result.data}")
    
    async def start_session_listener(self, callback: Callable[[Dict], None]):
        """
        Start polling for session changes (Fallback since Realtime is tricky in sync client).
        
        Args:
            callback: Function to call when a command is received
        """
        self.session_command_callback = callback
        logger.info("Starting session polling (interval: 5s)...")
        self.session_listener_task = asyncio.create_task(self._poll_sessions())

    async def _poll_sessions(self):
        """Poll database for active sessions."""
        last_status = None
        last_session_id = None
        
        while True:
            try:
                # Poll for the latest active session
                response = await asyncio.to_thread(
                    lambda: self.supabase.table("monitoring_sessions")
                    .select("*")
                    .eq("status", "active")
                    .order("created_at", desc=True)
                    .limit(1)
                    .execute()
                )
                
                data = response.data
                
                if data and len(data) > 0:
                    session = data[0]
                    session_id = session.get("id")
                    status = session.get("status")
                    config = session.get("config", {})
                    
                    # If we found a new active session we haven't seen before
                    if session_id != last_session_id:
                        logger.info(f"Detected active session: {session_id}")
                        self.current_session_id = session_id
                        if asyncio.iscoroutinefunction(self.session_command_callback):
                            await self.session_command_callback({
                                "action": "start",
                                "session_id": session_id,
                                "config": config
                            })
                        else:
                            self.session_command_callback({
                                "action": "start",
                                "session_id": session_id,
                                "config": config
                            })
                        last_session_id = session_id
                        last_status = status
                else:
                    # No active session found
                    if last_status == "active":
                        logger.info("Session stopped")
                        self.current_session_id = None
                        if asyncio.iscoroutinefunction(self.session_command_callback):
                            await self.session_command_callback({
                                "action": "stop",
                                "session_id": last_session_id
                            })
                        else:
                            self.session_command_callback({
                                "action": "stop",
                                "session_id": last_session_id
                            })
                        last_status = "stopped"
                        last_session_id = None
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
            
            await asyncio.sleep(5)  # Poll every 5 seconds
    
    async def stop(self):
        """Stop cloud sync service."""
        if self.session_listener_task:
            self.session_listener_task.cancel()
            try:
                await self.session_listener_task
            except asyncio.CancelledError:
                pass
        
        # Gracefully update active session status if one exists
        try:
            # We need to access the current active session ID from somewhere.
            # Since CloudSync tracks it via callbacks, we might not have it stored directly.
            # However, the violation engine has it.
            # Ideally, CloudSync should track the session it 'started'.
            # Let's modify _poll_sessions to store the current session ID in self.current_session_id
            if getattr(self, 'current_session_id', None):
                logger.info(f"Gracefully stopping session {self.current_session_id}...")
                await asyncio.to_thread(
                    lambda: self.supabase.table("monitoring_sessions")
                    .update({"status": "stopped"})
                    .eq("id", self.current_session_id)
                    .execute()
                )
        except Exception as e:
             logger.error(f"Failed to update session status on stop: {e}")

        logger.info("Cloud sync service stopped")
