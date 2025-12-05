"""
AI Client Service

Runs local object detection using YOLO models in a cascade pattern.
Directly integrates the model logic instead of calling an external API.
"""

import asyncio
import logging
import os
from typing import List, Dict
import numpy as np
from core import Config

logger = logging.getLogger(__name__)

# Try to import ultralytics
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    logger.warning("ultralytics not installed. Running in MOCK mode.")
    HAS_YOLO = False

class AIClient:
    """Client for running local object detection models."""
    
    def __init__(self, config: Config):
        self.config = config
        self.models = {}
        self._load_models()

    def _load_models(self):
        """Load all required YOLO models."""
        if not HAS_YOLO:
            logger.warning("YOLO not available, skipping model loading.")
            return

        # Model configuration
        # Assumes models are in a 'models' directory in the current working directory
        model_files = {
            "person": "Person_Test.pt",
            "eyes": "Eye_Test.pt", 
            "gloves": "Glove_Test.pt",
            "goggles": "Goggles_Test.pt",
            "hand": "Hand_Test.pt",
            "coat": "Lab_Coat_Test.pt"
        }

        for name, filename in model_files.items():
            try:
                # Check valid paths
                paths_to_check = [
                    os.path.join("models", filename),
                    filename,
                    os.path.join(os.getcwd(), "models", filename)
                ]
                
                model_path = None
                for p in paths_to_check:
                    if os.path.exists(p):
                        model_path = p
                        break
                
                if model_path:
                    logger.info(f"Loading model {name} from {model_path}...")
                    self.models[name] = YOLO(model_path)
                    # Log classes for verification
                    logger.info(f"Model {name} detects classes: {self.models[name].names}")
                else:
                    logger.warning(f"Model file {filename} not found. {name} detection will be skipped.")
            except Exception as e:
                logger.error(f"Failed to load model {name}: {e}")

    async def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Run cascade detection on the frame.
        
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
        # Handle Mock Mode or Missing YOLO
        if self.config.use_mock_detector or (not self.models and not HAS_YOLO):
            return self._mock_detect(frame)

        # Run detection in thread pool to avoid blocking async loop
        return await asyncio.to_thread(self._run_cascade_detection, frame)

    def _run_cascade_detection(self, frame: np.ndarray) -> List[Dict]:
        """
        Synchronous implementation of the cascade logic matching the requested flow:
        Layer 1: Person Detection -> Crop Person
        Layer 2:
             - Hand Detection -> Crop Hand -> Glove Detection
             - Eyes Detection -> Crop Eyes -> Goggles Detection
             - Lab Coat Detection
        """
        detections = []
        
        if "person" not in self.models:
            return []

        try:
            results_person = self.models["person"](frame, verbose=False)
        except Exception as e:
            logger.error(f"Error running person model: {e}")
            return []

        for result in results_person:
            boxes = result.boxes
            # Temporary Debug Log
            if len(boxes) > 0:
                logger.info(f"AI Client: Person Model found {len(boxes)} potential people.")
            else:
                # logger.info("AI Client: Person Model found 0 people.") # Uncomment to spam logs
                pass

            for box in boxes:
                # Get Person Box
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.models["person"].names.get(cls_id, "Person")
                logger.info(f"Detected {cls_name} at {conf:.2f}")

                # Add Person Detection
                detections.append({
                    "class": cls_name,
                    "bbox": [x1, y1, x2, y2],
                    "confidence": conf
                })
                logger.info(f"AI Client: Found {cls_name} (conf: {conf:.2f})")

                if conf < 0.4:
                    continue

                # Crop Person
                x1_c, y1_c = int(max(0, x1)), int(max(0, y1))
                x2_c, y2_c = int(min(frame.shape[1], x2)), int(min(frame.shape[0], y2))
                
                if x2_c <= x1_c or y2_c <= y1_c:
                    continue
                    
                person_crop = frame[y1_c:y2_c, x1_c:x2_c]
                
                # If crop is too small, skip sub-models
                if person_crop.shape[0] < 10 or person_crop.shape[1] < 10:
                    continue

                # ================= LAYER 2: Hand, Eyes, Coat =================
                
                # 1. Lab Coat (Directly on Person Crop)
                if "coat" in self.models:
                    try:
                        coat_results = self.models["coat"](person_crop, verbose=False)
                        for c_res in coat_results:
                            for c_box in c_res.boxes:
                                cx1, cy1, cx2, cy2 = c_box.xyxy[0].tolist()
                                c_conf = float(c_box.conf[0])
                                c_cls_id = int(c_box.cls[0])
                                c_cls_name = self.models["coat"].names.get(c_cls_id, "Lab_Coat")
                                
                                # Global Coords: Coat_Local + Person_Offset
                                gx1 = cx1 + x1_c
                                gy1 = cy1 + y1_c
                                gx2 = cx2 + x1_c
                                gy2 = cy2 + y1_c
                                
                                detections.append({
                                    "class": c_cls_name,
                                    "bbox": [gx1, gy1, gx2, gy2],
                                    "confidence": c_conf
                                })
                                logger.info(f"AI Client: Found {c_cls_name} (conf: {c_conf:.2f})")
                    except Exception as e:
                        logger.error(f"Error running coat model: {e}")

                # 2. Hand Detection -> Crop -> Glove Detection
                if "hand" in self.models:
                    try:
                        hand_results = self.models["hand"](person_crop, verbose=False)
                        for h_res in hand_results:
                            for h_box in h_res.boxes:
                                hx1, hy1, hx2, hy2 = h_box.xyxy[0].tolist()
                                h_conf = float(h_box.conf[0])
                                h_cls_id = int(h_box.cls[0])
                                h_cls_name = self.models["hand"].names.get(h_cls_id, "Hand")
                                
                                # Add Hand detection (Optional, useful for debugging)
                                detections.append({
                                    "class": h_cls_name,
                                    "bbox": [hx1 + x1_c, hy1 + y1_c, hx2 + x1_c, hy2 + y1_c],
                                    "confidence": h_conf
                                })

                                if h_conf < 0.4:
                                    continue
                                
                                # Crop Hand
                                hx1_c, hy1_c = int(max(0, hx1)), int(max(0, hy1))
                                hx2_c, hy2_c = int(min(person_crop.shape[1], hx2)), int(min(person_crop.shape[0], hy2))
                                
                                if hx2_c <= hx1_c or hy2_c <= hy1_c:
                                    continue
                                
                                hand_crop = person_crop[hy1_c:hy2_c, hx1_c:hx2_c]
                                
                                # Run Glove Model on Hand Crop
                                if "gloves" in self.models and hand_crop.shape[0] > 5 and hand_crop.shape[1] > 5:
                                    try:
                                        glove_results = self.models["gloves"](hand_crop, verbose=False)
                                        for g_res in glove_results:
                                            for g_box in g_res.boxes:
                                                gx1, gy1, gx2, gy2 = g_box.xyxy[0].tolist()
                                                g_conf = float(g_box.conf[0])
                                                g_cls_id = int(g_box.cls[0])
                                                g_cls_name = self.models["gloves"].names.get(g_cls_id, "Gloves")
                                                
                                                # Global Coords: Glove_Local + Hand_Offset + Person_Offset
                                                final_gx1 = gx1 + hx1_c + x1_c
                                                final_gy1 = gy1 + hy1_c + y1_c
                                                final_gx2 = gx2 + hx1_c + x1_c
                                                final_gy2 = gy2 + hy1_c + y1_c
                                
                                detections.append({
                                    "class": g_cls_name,
                                    "bbox": [final_gx1, final_gy1, final_gx2, final_gy2],
                                    "confidence": g_conf
                                })
                                logger.info(f"AI Client: Found {g_cls_name} (conf: {g_conf:.2f})")
                    except Exception as e:
                                        logger.error(f"Error running glove model: {e}")

                    except Exception as e:
                        logger.error(f"Error running hand model: {e}")

                # 3. Eyes Detection -> Crop -> Goggles Detection
                if "eyes" in self.models:
                    try:
                        eyes_results = self.models["eyes"](person_crop, verbose=False)
                        for e_res in eyes_results:
                            for e_box in e_res.boxes:
                                ex1, ey1, ex2, ey2 = e_box.xyxy[0].tolist()
                                e_conf = float(e_box.conf[0])
                                e_cls_id = int(e_box.cls[0])
                                e_cls_name = self.models["eyes"].names.get(e_cls_id, "Eyes")
                                
                                # Add Eyes detection
                                detections.append({
                                    "class": e_cls_name,
                                    "bbox": [ex1 + x1_c, ey1 + y1_c, ex2 + x1_c, ey2 + y1_c],
                                    "confidence": e_conf
                                })

                                if e_conf < 0.4:
                                    continue
                                
                                # Crop Eyes
                                ex1_c, ey1_c = int(max(0, ex1)), int(max(0, ey1))
                                ex2_c, ey2_c = int(min(person_crop.shape[1], ex2)), int(min(person_crop.shape[0], ey2))
                                
                                if ex2_c <= ex1_c or ey2_c <= ey1_c:
                                    continue
                                
                                eyes_crop = person_crop[ey1_c:ey2_c, ex1_c:ex2_c]
                                
                                # Run Goggles Model on Eyes Crop
                                if "goggles" in self.models and eyes_crop.shape[0] > 5 and eyes_crop.shape[1] > 5:
                                    try:
                                        goggles_results = self.models["goggles"](eyes_crop, verbose=False)
                                        for g_res in goggles_results:
                                            for g_box in g_res.boxes:
                                                ggx1, ggy1, ggx2, ggy2 = g_box.xyxy[0].tolist()
                                                gg_conf = float(g_box.conf[0])
                                                gg_cls_id = int(g_box.cls[0])
                                                gg_cls_name = self.models["goggles"].names.get(gg_cls_id, "Goggles")
                                                
                                                # Global Coords: Goggles_Local + Eyes_Offset + Person_Offset
                                                final_ggx1 = ggx1 + ex1_c + x1_c
                                                final_ggy1 = ggy1 + ey1_c + y1_c
                                                final_ggx2 = ggx2 + ex1_c + x1_c
                                                final_ggy2 = ggy2 + ey1_c + y1_c
                                                
                                                detections.append({
                                                    "class": gg_cls_name,
                                                    "bbox": [final_ggx1, final_ggy1, final_ggx2, final_ggy2],
                                                    "confidence": gg_conf
                                                })
                                    except Exception as e:
                                        logger.error(f"Error running goggles model: {e}")

                    except Exception as e:
                        logger.error(f"Error running eyes model: {e}")

        return detections

    def _mock_detect(self, frame: np.ndarray) -> List[Dict]:
        """Return mock detections."""
        h, w = frame.shape[:2]
        x1 = int(w * 0.3)
        y1 = int(h * 0.2)
        x2 = int(w * 0.7)
        y2 = int(h * 0.8)
        
        # Mock Person + Goggles
        return [
            {
                "class": "Person",
                "bbox": [x1, y1, x2, y2],
                "confidence": 0.95
            },
            {
                "class": "Goggles",
                "bbox": [int(w*0.45), int(h*0.25), int(w*0.55), int(h*0.3)],
                "confidence": 0.95
            }
        ]
    
    async def health_check(self) -> bool:
        """Check if models are loaded."""
        if self.config.use_mock_detector:
            return True
        return len(self.models) > 0
    
    async def close(self):
        """Cleanup resources."""
        self.models.clear()
