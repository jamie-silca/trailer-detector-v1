# Trailer Detector Service v1

A lightweight, Dockerized microservice for detecting trailer IDs using YOLOv11n.

## Features

- **FastAPI** backend with async support.
- **YOLOv11n** inference optimized for CPU.
- **Single & Batch** image processing.
- **Dockerized** for easy deployment.
- **Swagger/OpenAPI** documentation auto-generated.

## Setup

1. **Prerequisites**:
   - Docker and Docker Compose installed.
   - Model file downloaded and placed in `model/` directory.
     - **RT-DETR**: [HuggingFace - trailerIdDetect-rd](https://huggingface.co/flightsnotights/trailerIdDetect-rd/tree/main)
     - **YOLOv11n**: [HuggingFace - trailerIdDetect-y11n](https://huggingface.co/flightsnotights/trailerIdDetect-y11n/tree/main)

2. **Model Installation**:
   - Ensure `model/yolov11n.pt` exists.

## Running the Service

Build and start the service:

```bash
docker-compose up --build -d
```

The service will start on port **8000**.

## API Documentation

Once running, visit:
- **Swagger UI**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **ReDoc**: [http://localhost:8000/redoc](http://localhost:8000/redoc)

## Usage Examples

### Health Check

```bash
curl http://localhost:8000/health
```

### Detect Single Image

```bash
curl -X POST "http://localhost:8000/detect?confidence_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

### Detect Batch Images

```bash
curl -X POST "http://localhost:8000/detect/batch?confidence_threshold=0.5" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "files=@/path/to/image1.jpg" \
  -F "files=@/path/to/image2.jpg"
```
