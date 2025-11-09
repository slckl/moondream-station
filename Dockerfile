# Moondream Station Non-Interactive Server
#
# Build:
#   docker build -t moondream-station .
#
# Run:
#   docker run --gpus all -p 2020:2020 moondream-station
#
# Test:
#   curl http://localhost:2020/health

FROM nvidia/cuda:12.6.0-base-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive

# Install prerequisites including Python
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    tar \
    bash \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV PATH="/root/.local/bin/:$PATH"

# Set working directory
WORKDIR /moondream-station

# Copy project files
COPY . .

# Sync dependencies and install moondream_station
RUN uv sync && uv pip install moondream_station

# Run non-interactive setup to:
# - Install PyTorch with CUDA 12.6 support
# - Download moondream2 model
# - Setup backend requirements
RUN uv run non_interactive_setup.py

# Expose the default API server port
EXPOSE 2020

# Set default command to run non-interactive server
CMD ["uv", "run", "non_interactive_server.py"]
