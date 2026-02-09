# Use Ultralytics base image which includes PyTorch and dependencies
FROM ultralytics/ultralytics:latest-cpu

# Set working directory
WORKDIR /app

# Install system dependencies if needed (Ultralytics image has most, but let's be safe for minimal ones if missing, usually not needed for this stack)
# We can skip apt-get for now unless we know something is missing.


# Copy requirements
COPY requirements.txt .

# Install dependencies
# --no-cache-dir to keep image small
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app /app/app

# Create directory for model mount
RUN mkdir -p /app/model

# Create a non-root user
RUN useradd -m appuser && chown -R appuser /app
USER appuser

# Expose port
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
