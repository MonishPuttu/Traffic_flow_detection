
# Use for production (longer build time)
# FROM python:3.10-slim

# WORKDIR /app

# # Install system deps
# RUN apt-get update && apt-get install -y \
#     git ffmpeg libsm6 libxext6 && rm -rf /var/lib/apt/lists/*

# # Copy and install Python deps
# COPY requirements.txt .
# RUN pip install --no-cache-dir -r requirements.txt

# # Copy source code
# COPY . /app

# EXPOSE 8000

# # Run the FastAPI app
# CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]



# Use for testing (faster builds)
FROM ultralytics/ultralytics:latest

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
