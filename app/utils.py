import io
import requests
from PIL import Image
from datetime import datetime, timezone

def bytes_to_image(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    try:
        image = Image.open(io.BytesIO(image_bytes))
        # Ensure image is in RGB mode (handle RGBA, P, etc.)
        if image.mode != "RGB":
            image = image.convert("RGB")
        return image
    except Exception as e:
        raise ValueError(f"Invalid image data: {str(e)}")

def download_image_from_url(url: str) -> Image.Image:
    """Download image from URL and return PIL Image."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        return bytes_to_image(response.content)
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Failed to download image from {url}: {str(e)}")
    except Exception as e:
        raise ValueError(f"Error processing image from URL: {str(e)}")

def get_current_timestamp() -> datetime:
    """Get current UTC timestamp."""
    return datetime.now(timezone.utc)

def format_timestamp_iso(dt: datetime) -> str:
    """Format timestamp to ISO 8601 string."""
    return dt.isoformat()
