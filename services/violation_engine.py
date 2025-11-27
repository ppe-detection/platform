"""
Violation Engine Service

Processes detections to identify PPE violations.
Implements person-PPE mapping, compliance checking, and debouncing.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime, timedelta
import cv2
import numpy as np

from core import Config
from .cloud_sync import CloudSync

logger = logging.getLogger(__name__)


class PersonTracker:
    """Tracks a person and their associated PPE."""
    
    def __init__(self, person_id: str, bbox: List[float]):
        self.person_id = person_id
        self.bbox = bbox  # [x1, y1, x2, y2]
        self.ppe: Dict[str, List[Dict]] = defaultdict(list)  # class -> [detections]
        self.last_seen = datetime.now()
    
    def update_ppe(self, detections: List[Dict]):
        """Update PPE detections for this person."""
        self.ppe.clear()
        for det in detections:
            ppe_class = det.get("class", "").lower()
            self.ppe[ppe_class].append(det)
        self.last_seen = datetime.now()
    
    def has_ppe(self, required_class: str) -> bool:
        """Check if person has specific PPE."""
        return required_class.lower() in self.ppe and len(self.ppe[required_class.lower()]) > 0


class ViolationEngine:
    """Engine for detecting and managing PPE violations."""
    
    def __init__(self, config: Config, cloud_sync: CloudSync):
        self.config = config
        self.cloud_sync = cloud_sync
        
        # Person tracking
        self.people: Dict[str, PersonTracker] = {}  # person_id -> PersonTracker
        
        # Violation tracking (for debouncing)
        self.active_violations: Dict[str, Dict] = {}  # violation_key -> violation_data
        self.violation_start_times: Dict[str, datetime] = {}  # violation_key -> start_time
        self.violation_cooldowns: Dict[str, datetime] = {}  # violation_key -> cooldown_until
        
        # Active session
        self.active_session_id: Optional[str] = None
        
        # Required PPE (can be updated from Supabase)
        self.required_ppe = {
            "goggles": config.require_goggles,
            "lab_coat": config.require_lab_coat,
            "gloves": config.require_gloves,
            "helmet": config.require_helmet,
        }
    
    def set_active_session(self, session_id: str, config: Dict = None):
        """Set the active lab session and update configuration."""
        self.active_session_id = session_id
        logger.info(f"Active session set to: {session_id}")
        
        if config:
            self.update_config(config)
            
    def update_config(self, config: Dict):
        """Update PPE requirements from session config."""
        logger.info(f"Updating PPE requirements from config: {config}")
        
        # Map config keys to required_ppe keys
        # Spec config format: {"goggles_enabled": true, "lab_coat_enabled": true, "gloves_enabled": true}
        
        if "goggles_enabled" in config:
            self.required_ppe["goggles"] = config.get("goggles_enabled", False)
            
        if "lab_coat_enabled" in config:
            self.required_ppe["lab_coat"] = config.get("lab_coat_enabled", False)
            
        if "gloves_enabled" in config:
            self.required_ppe["gloves"] = config.get("gloves_enabled", False)
            
        # Handle potential mapping differences or additional keys
        if "helmet_enabled" in config:
             self.required_ppe["helmet"] = config.get("helmet_enabled", False)

        logger.info(f"New PPE requirements: {self.required_ppe}")
    
    def clear_active_session(self):
        """Clear the active lab session."""
        self.active_session_id = None
        logger.info("Active session cleared")
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two bounding boxes."""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def _match_ppe_to_person(
        self,
        person_bbox: List[float],
        ppe_detections: List[Dict]
    ) -> List[Dict]:
        """Match PPE detections to a person based on IoU."""
        matched_ppe = []
        
        for ppe_det in ppe_detections:
            ppe_bbox = ppe_det.get("bbox", [])
            if not ppe_bbox:
                continue
            
            iou = self._calculate_iou(person_bbox, ppe_bbox)
            if iou >= self.config.iou_threshold:
                matched_ppe.append(ppe_det)
        
        return matched_ppe
    
    def _generate_person_id(self, bbox: List[float], camera_id: str) -> str:
        """Generate a stable person ID based on bbox center and camera."""
        # Use bbox center as identifier (in production, use proper tracking)
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # Simple ID based on position (could be improved with proper tracking)
        return f"{camera_id}_{int(center_x)}_{int(center_y)}"
    
    async def process_detections(
        self,
        camera_id: str,
        frame: np.ndarray,
        detections: List[Dict]
    ):
        """
        Process detections to identify violations.
        
        Args:
            camera_id: ID of the camera
            frame: Current frame (for snapshots)
            detections: List of detections from AI model
        """
        if not self.active_session_id:
            # No active session, skip processing
            return
        
        # Separate people and PPE
        people_detections = [
            d for d in detections
            if d.get("class", "").lower() == "person"
        ]
        ppe_detections = [
            d for d in detections
            if d.get("class", "").lower() in ["goggles", "lab_coat", "gloves", "helmet"]
        ]
        
        # Update person tracking
        current_people = {}
        for person_det in people_detections:
            person_bbox = person_det.get("bbox", [])
            if not person_bbox:
                continue
            
            person_id = self._generate_person_id(person_bbox, camera_id)
            
            # Match PPE to this person
            matched_ppe = self._match_ppe_to_person(person_bbox, ppe_detections)
            
            # Update or create person tracker
            if person_id in self.people:
                tracker = self.people[person_id]
                tracker.bbox = person_bbox
                tracker.update_ppe(matched_ppe)
            else:
                tracker = PersonTracker(person_id, person_bbox)
                tracker.update_ppe(matched_ppe)
                self.people[person_id] = tracker
            
            current_people[person_id] = tracker
        
        # Remove old people (not seen in this frame)
        now = datetime.now()
        expired_people = [
            pid for pid, tracker in self.people.items()
            if pid not in current_people and
            (now - tracker.last_seen).total_seconds() > 5.0
        ]
        for pid in expired_people:
            del self.people[pid]
        
        # Check for violations
        for person_id, tracker in current_people.items():
            await self._check_violations(camera_id, person_id, tracker, frame)
    
    async def _check_violations(
        self,
        camera_id: str,
        person_id: str,
        tracker: PersonTracker,
        frame: np.ndarray
    ):
        """Check for PPE violations for a specific person."""
        now = datetime.now()
        
        # Check each required PPE
        for ppe_class, required in self.required_ppe.items():
            if not required:
                continue
            
            violation_key = f"{camera_id}_{person_id}_{ppe_class}"
            
            # Check if in cooldown
            if violation_key in self.violation_cooldowns:
                if now < self.violation_cooldowns[violation_key]:
                    continue  # Still in cooldown
                else:
                    del self.violation_cooldowns[violation_key]
            
            # Check if person has required PPE
            has_ppe = tracker.has_ppe(ppe_class)
            
            if not has_ppe:
                # Violation detected
                if violation_key not in self.violation_start_times:
                    # New violation, record start time
                    self.violation_start_times[violation_key] = now
                
                # Check if violation has persisted long enough (debouncing)
                violation_duration = (
                    now - self.violation_start_times[violation_key]
                ).total_seconds()
                
                if violation_duration >= self.config.violation_debounce_seconds:
                    # Violation confirmed, trigger alert
                    if violation_key not in self.active_violations:
                        await self._trigger_violation_alert(
                            camera_id,
                            person_id,
                            tracker,
                            ppe_class,
                            frame
                        )
            else:
                # Person has required PPE, clear violation
                if violation_key in self.violation_start_times:
                    del self.violation_start_times[violation_key]
                if violation_key in self.active_violations:
                    del self.active_violations[violation_key]
    
    async def _trigger_violation_alert(
        self,
        camera_id: str,
        person_id: str,
        tracker: PersonTracker,
        missing_ppe: str,
        frame: np.ndarray
    ):
        """Trigger a violation alert and upload to Supabase."""
        violation_key = f"{camera_id}_{person_id}_{missing_ppe}"
        
        logger.warning(
            f"Violation detected: Person {person_id} missing {missing_ppe} "
            f"(Camera: {camera_id})"
        )
        
        # Extract person region from frame
        x1, y1, x2, y2 = [int(coord) for coord in tracker.bbox]
        # Add padding
        padding = 20
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame.shape[1], x2 + padding)
        y2 = min(frame.shape[0], y2 + padding)
        
        person_frame = frame[y1:y2, x1:x2]
        
        # Mark as active violation
        self.active_violations[violation_key] = {
            "camera_id": camera_id,
            "person_id": person_id,
            "missing_ppe": missing_ppe,
            "timestamp": datetime.now(),
            "frame": person_frame,
            "bbox": tracker.bbox
        }
        
        # Set cooldown
        self.violation_cooldowns[violation_key] = (
            datetime.now() + timedelta(seconds=self.config.violation_cooldown_seconds)
        )
        
        # Upload to Supabase
        await self.cloud_sync.upload_violation(
            session_id=self.active_session_id,
            camera_id=camera_id,
            person_id=person_id,
            missing_ppe=missing_ppe,
            frame=person_frame,
            bbox=tracker.bbox
        )
