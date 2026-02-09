import os
import time
import logging
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image
from ultralytics import YOLO, RTDETR

from app.models import Detection, BoundingBox, DetectionResponse, ImageMetadata
from app.utils import get_current_timestamp

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class YoloDetector:
    _instance = None
    _model = None
    
    # Path inside container (will be mounted)
    DEFAULT_MODEL_PATH = "/app/model/rtdetr_r18.pth"

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(YoloDetector, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance

    def _initialize(self):
        """Initialize the YOLO model."""
        model_path = os.getenv("YOLO_MODEL_PATH", self.DEFAULT_MODEL_PATH)
        logger.info(f"Loading YOLO model from {model_path}...")
        
        try:
            # Load model
            if model_path.endswith('.pth'):
                logger.info("Detected RT-DETR model extension. Creating temporary .pt file for Ultralytics compatibility...")
                import shutil
                import tempfile
                
                # Create temp file with .pt extension
                temp_dir = tempfile.gettempdir()
                temp_path = os.path.join(temp_dir, "temp_model.pt")
                shutil.copy2(model_path, temp_path)
                logger.info(f"Copied model to {temp_path}")
                
                try:
                    self._model = RTDETR(temp_path)
                    logger.info("RT-DETR model loaded successfully from temporary file.")
                finally:
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                        logger.info("Temporary model file removed.")
            else:
                self._model = YOLO(model_path)
                
            # Force CPU
            self._model.to("cpu")
            logger.info("Model loaded successfully on CPU.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Could not load model at {model_path}: {e}")

    def detect(self, image: Image.Image, filename: str, conf_threshold: float = 0.0) -> DetectionResponse:
        """
        Run inference on a single image.
        """
        start_time = time.perf_counter()
        timestamp = get_current_timestamp()
        
        try:
            # Run inference
            # conf=conf_threshold filters by confidence
            # verbose=False reduces log spam
            results = self._model.predict(image, conf=conf_threshold, device="cpu", verbose=False)
            result = results[0] # Single image
            
            detections = []
            
            for box in result.boxes:
                # Get box coordinates (x1, y1, x2, y2)
                xyxy = box.xyxy[0].tolist()
                x1, y1, x2, y2 = xyxy
                
                # Calculate width/height
                w = x2 - x1
                h = y2 - y1
                
                # Get confidence and class
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                # Class name (assuming trailer_id is class 0 or mapping names from model)
                # If model has names, use them, otherwise fallback
                cls_name = result.names.get(cls_id, "unknown")

                # Normalize Bounding Box
                img_w = float(image.width)
                img_h = float(image.height)
                
                detection = Detection(
                    class_id=cls_id,
                    class_name=cls_name,
                    confidence=conf,
                    bbox=BoundingBox(
                        x_min=x1 / img_w,
                        y_min=y1 / img_h,
                        x_max=x2 / img_w,
                        y_max=y2 / img_h,
                        width=w / img_w,
                        height=h / img_h
                    )
                )
                detections.append(detection)
            
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return DetectionResponse(
                success=True,
                timestamp=timestamp,
                processing_time_ms=round(processing_time, 2),
                image_metadata=ImageMetadata(
                    filename=filename,
                    width=image.width,
                    height=image.height
                ),
                detections=detections,
                detection_count=len(detections)
            )
            
        except Exception as e:
            logger.error(f"Detection failed for {filename}: {e}")
            processing_time = (time.perf_counter() - start_time) * 1000
            
            return DetectionResponse(
                success=False,
                timestamp=timestamp,
                processing_time_ms=round(processing_time, 2),
                image_metadata=ImageMetadata(
                    filename=filename,
                    width=image.width,
                    height=image.height
                ),
                detections=[],
                detection_count=0,
                error=str(e)
            )

    def detect_batch(self, images: List[Tuple[Image.Image, str]], conf_threshold: float = 0.0) -> List[DetectionResponse]:
        """
        Run inference on a batch of images in parallel.
        """
        # Since YOLO predict can accept a list of images, we could use that,
        # but handling individual filenames and mapping results back is often easier with mapping.
        # However, calling model.predict(list_of_images) is usually more efficient for batching if model supports it.
        # But for valid mapping of "Filename -> Result", and given this is a lightweight CPU service,
        # parallel execution of the `detect` method via threadpool might be safer for individual error handling
        # and simplicity, although slightly less performant than true batch inference.
        # Let's try true batch inference first for performance, as requested "fast".
        
        # Actually, ultralytics supports list of images.
        # Let's separate PIL images and filenames
        pil_images = [img for img, _ in images]
        filenames = [name for _, name in images]
        
        start_time = time.perf_counter()
        timestamp = get_current_timestamp()
        
        responses = []
        
        try:
            # True batch inference
            results = self._model.predict(pil_images, conf=conf_threshold, device="cpu", verbose=False)
            
            end_time = time.perf_counter() 
            # We can't easily attribute per-image time in batch, so we'll approximate or just use total time split?
            # Actually, let's just use the total time / count for reporting, or just report total batch time in wrapper.
            # The DetectResponse expects processing_time_ms.
            
            for i, result in enumerate(results):
                filename = filenames[i]
                image = pil_images[i]
                detections = []
                
                # Image dimensions for normalization
                img_w = float(image.width)
                img_h = float(image.height)

                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = xyxy
                    w = x2 - x1
                    h = y2 - y1
                    conf = float(box.conf[0])
                    cls_id = int(box.cls[0])
                    cls_name = result.names.get(cls_id, "unknown")
                    
                    detections.append(Detection(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=conf,
                        bbox=BoundingBox(
                            x_min=x1 / img_w,
                            y_min=y1 / img_h,
                            x_max=x2 / img_w,
                            y_max=y2 / img_h,
                            width=w / img_w,
                            height=h / img_h
                        )
                    ))
                
                responses.append(DetectionResponse(
                    success=True,
                    timestamp=timestamp,
                    processing_time_ms=round(((end_time - start_time) * 1000) / len(results), 2), # Averaged
                    image_metadata=ImageMetadata(
                        filename=filename,
                        width=image.width,
                        height=image.height
                    ),
                    detections=detections,
                    detection_count=len(detections)
                ))
                
        except Exception as e:
            logger.error(f"Batch detection failed: {e}")
            # Fallback: try serial or return errors
            for img, fname in images:
                 responses.append(DetectionResponse(
                    success=False,
                    timestamp=timestamp,
                    processing_time_ms=0,
                    image_metadata=ImageMetadata(filename=fname, width=img.width, height=img.height),
                    detections=[],
                    detection_count=0,
                    error=f"Batch processing failed: {str(e)}"
                ))

        return responses
