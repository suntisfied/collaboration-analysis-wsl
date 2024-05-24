# Use the official NVIDIA CUDA runtime base image
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Install ffmpeg, git, and Python
RUN apt-get update && apt-get install -y \
    ffmpeg \
    git \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1

# Set the working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip3 install --upgrade pip

# Install PyTorch, torchvision, and torchaudio with CUDA support
RUN pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu121

# Install the remaining dependencies
RUN pip3 install -r requirements.txt

# Clone and install whisper from GitHub
RUN git clone https://github.com/openai/whisper.git /tmp/whisper \
    && cd /tmp/whisper \
    && pip3 install .

# Copy the rest of the application code
COPY . /app/

# Set the environment variable for Hugging Face token
ENV HUGGINGFACE_TOKEN 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

# Set Environment Variables
ENV CUDA_VISIBLE_DEVICES=0

# Set the entrypoint
ENTRYPOINT ["python3", "text_extractor.py"]
