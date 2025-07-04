# Use official NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.3.2-base-ubuntu22.04

# Create python alias (optional, if your script uses `python`)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Set working directory inside the container
WORKDIR /app

# Copy project files (excluding cache/ and venv/ via .dockerignore)
COPY . .

# Install system dependencies and Python
RUN apt-get update && \
    apt-get install -y python3 python3-pip git && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --upgrade pip && \
    pip3 install --no-cache-dir .

# Make the entrypoint script executable
RUN chmod +x /app/run.sh

# Set default environment variables
ENV HF_TOKEN=missing
ENV CUDA_VISIBLE_DEVICES=0
ENV CACHE_DIR=/app/cache
ENV INPUT_DIR=/app/inputs
ENV OUTPUT_DIR=/app/output
ENV INPUT_FILENAME=sample_interview.txt
ENV INTRO_PROMPT="Jonathan Carker interviewing Cheryl Jones on 30th September at Grand Union's magnificent Bothy."

# Default entrypoint forwards user arguments to the script
ENTRYPOINT /app/run.sh "$INPUT_FILENAME" "$INTRO_PROMPT"

