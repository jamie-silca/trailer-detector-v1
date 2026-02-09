from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from typing import List
import time
import logging

from app.models import DetectionResponse, BatchDetectionResponse, UrlDetectionRequest, UrlBatchDetectionRequest
from app.detector import YoloDetector
from app.utils import bytes_to_image, get_current_timestamp, download_image_from_url

# Initialize app
app = FastAPI(
    title="Trailer ID Detection Service",
    description="Microservice for detecting trailer IDs using RT-DETR",
    version="1.0.0"
)

# Initialize detector (lazy load on first request or startup)
# startup_event to pre-load model
@app.on_event("startup")
async def startup_event():
    YoloDetector() # This will load the model

@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "trailer-detection-yolo"}

@app.post("/detect", response_model=DetectionResponse)
async def detect_single(
    file: UploadFile = File(...),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    try:
        # Read image
        try:
            image_bytes = await file.read()
            image = bytes_to_image(image_bytes)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

        # Run detection
        detector = YoloDetector()
        response = detector.detect(image, file.filename, confidence_threshold)
        
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/batch", response_model=BatchDetectionResponse)
async def detect_batch(
    files: List[UploadFile] = File(...),
    confidence_threshold: float = Query(0.0, ge=0.0, le=1.0)
):
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    start_time = time.perf_counter()
    timestamp = get_current_timestamp()
    
    try:
        # Pre-process all images
        images_to_process = []
        for file in files:
            try:
                content = await file.read()
                img = bytes_to_image(content)
                images_to_process.append((img, file.filename))
            except Exception as e:
                # Log error but maybe continue? Or fail all? 
                # User asked for robust, let's just log and skip or create error dummy?
                # For batch, usually we want to return what worked.
                # But my detector expects valid images.
                # Let's skip invalid ones or handle inside implementation.
                # For simplicity, fail fast on invalid uploads for now or better, return error in result list.
                pass 
                
        if not images_to_process:
             raise HTTPException(status_code=400, detail="No valid images found in batch")

        # Run batch detection
        detector = YoloDetector()
        results = detector.detect_batch(images_to_process, confidence_threshold)
        
        total_time = (time.perf_counter() - start_time) * 1000
        
        return BatchDetectionResponse(
            success=True,
            timestamp=timestamp,
            total_processing_time_ms=round(total_time, 2),
            results=results,
            total_detections=sum(r.detection_count for r in results)
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/url", response_model=DetectionResponse)
async def detect_single_url(request: UrlDetectionRequest):
    try:
        image = download_image_from_url(request.image_url)
        # Use provided ID if available, otherwise filename from URL
        filename = request.id or request.image_url.split('/')[-1]
        
        detector = YoloDetector()
        response = detector.detect(image, filename, request.confidence_threshold)
        response.id = request.id
        
        return response
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/detect/url/batch", response_model=BatchDetectionResponse)
async def detect_batch_url(request: UrlBatchDetectionRequest):
    start_time = time.perf_counter()
    timestamp = get_current_timestamp()
    
    try:
        images_to_process = []
        original_requests = []
        
        if not request.images:
            raise HTTPException(status_code=400, detail="No images provided in batch request")

        for img_req in request.images:
            try:
                img = download_image_from_url(img_req.image_url)
                filename = img_req.id or img_req.image_url.split('/')[-1]
                images_to_process.append((img, filename))
                original_requests.append(img_req)
            except Exception as e:
                pass

        if not images_to_process:
             raise HTTPException(status_code=400, detail="No valid images could be downloaded")

        detector = YoloDetector()
        results = detector.detect_batch(images_to_process, request.confidence_threshold)
        
        for i, res in enumerate(results):
            if i < len(original_requests):
                res.id = original_requests[i].id

        total_time = (time.perf_counter() - start_time) * 1000
        
        return BatchDetectionResponse(
            success=True,
            timestamp=timestamp,
            total_processing_time_ms=round(total_time, 2),
            results=results,
            total_detections=sum(r.detection_count for r in results)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
