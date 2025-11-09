#!/usr/bin/env python3
"""
Non-interactive setup script for Moondream Station in Docker environments.
This script sets up CUDA backend and downloads the moondream2 model for inference.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import requests

DEFAULT_MANIFEST_URL = "https://m87-md-prod-assets.s3.us-west-2.amazonaws.com/station/mds2/production_manifest.json"
CUDA_VERSION = "12.6"
TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu126"


def log(message: str):
    """Print a log message."""
    print(f"[SETUP] {message}", flush=True)


def run_command(cmd: list[str], description: str):
    """Run a command and handle errors."""
    log(f"{description}...")
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        log(f"✓ {description} completed")
        return result
    except subprocess.CalledProcessError as e:
        log(f"✗ {description} failed")
        log(f"stdout: {e.stdout}")
        log(f"stderr: {e.stderr}")
        raise


def install_pytorch_and_deps():
    """Install PyTorch with CUDA 12.6 support and core ML dependencies."""
    log(f"Installing PyTorch with CUDA {CUDA_VERSION} support")

    # Install torch, torchvision, torchaudio with CUDA support
    cmd = [
        "uv",
        "pip",
        "install",
        "--extra-index-url",
        TORCH_INDEX_URL,
        "torch>=2.7.0",
        "torchvision",
        "torchaudio",
        "transformers>=4.56.1",
        "accelerate>=1.8.1",
        "Pillow>=9.0.0",
    ]

    run_command(cmd, "PyTorch and core ML dependencies installation")


def install_backend_requirements():
    """Install backend requirements from the manifest."""
    log("Fetching production manifest")

    try:
        response = requests.get(DEFAULT_MANIFEST_URL, timeout=30)
        response.raise_for_status()
        manifest_data = response.json()
        log("✓ Manifest fetched successfully")
    except Exception as e:
        log(f"✗ Failed to fetch manifest: {e}")
        raise

    # Get backend requirements
    backends = manifest_data.get("backends", {})

    for backend_id, backend_info in backends.items():
        log(f"Processing backend: {backend_id}")

        # Download and extract backend if it has a download URL
        download_url = backend_info.get("download_url")
        if download_url:
            log(f"Downloading backend from {download_url}")

            try:
                response = requests.get(download_url, timeout=60)
                response.raise_for_status()

                # Extract to backends directory
                backends_dir = Path("backends")
                backends_dir.mkdir(exist_ok=True)

                # Save tarball temporarily
                with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False) as f:
                    f.write(response.content)
                    tarball_path = f.name

                # Extract
                import tarfile

                with tarfile.open(tarball_path, "r:gz") as tar:
                    tar.extractall(path=backends_dir)

                Path(tarball_path).unlink()
                log(f"✓ Backend {backend_id} extracted")

                # Install requirements from the extracted backend
                backend_name = download_url.split("/")[-1].replace(".tar.gz", "")
                requirements_file = backends_dir / backend_name / "requirements.txt"

                if requirements_file.exists():
                    log(f"Installing requirements from {requirements_file}")

                    # Read requirements and install with CUDA support
                    with open(requirements_file) as f:
                        requirements_content = f.read()

                    # Create temporary requirements file
                    with tempfile.NamedTemporaryFile(
                        mode="w", suffix=".txt", delete=False
                    ) as f:
                        f.write(requirements_content)
                        temp_requirements = f.name

                    # Check if requirements include torch packages
                    has_torch = any(
                        pkg.startswith(("torch", "torchvision", "torchaudio"))
                        for pkg in requirements_content.lower().split("\n")
                    )

                    cmd = [
                        "uv",
                        "pip",
                        "install",
                        "-r",
                        temp_requirements,
                    ]

                    # Add torch index URL if torch packages are present
                    if has_torch:
                        cmd.extend(["--extra-index-url", TORCH_INDEX_URL])

                    run_command(cmd, f"Backend {backend_id} requirements installation")
                    Path(temp_requirements).unlink()
                else:
                    log(f"No requirements.txt found for {backend_id}")

            except Exception as e:
                log(f"✗ Failed to process backend {backend_id}: {e}")
                raise


def download_moondream2_model():
    """Pre-download the moondream2 model to avoid runtime downloads."""
    log("Pre-downloading moondream2 model")

    # Create a simple script to trigger model download
    download_script = """
import torch
from transformers import AutoModelForCausalLM
import os

# Set HuggingFace cache to a predictable location
cache_dir = os.path.expanduser("~/.cache/huggingface")
os.makedirs(cache_dir, exist_ok=True)

print("Downloading moondream2 model...")
model = AutoModelForCausalLM.from_pretrained(
    "vikhyatk/moondream2",
    trust_remote_code=True,
    cache_dir=cache_dir
)
print("Model downloaded successfully!")
print(f"Model config commit hash: {model.config._commit_hash}")
"""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(download_script)
        script_path = f.name

    try:
        cmd = ["uv", "run", script_path]
        run_command(cmd, "Moondream2 model download")
    finally:
        Path(script_path).unlink()


def create_config():
    """Create configuration file to store CUDA version."""
    log("Creating configuration")

    app_dir = Path.home() / ".moondream-station"
    app_dir.mkdir(exist_ok=True)

    config_file = app_dir / "config.json"
    config = {
        "torch_cuda_version": CUDA_VERSION,
        "torch_index_url": TORCH_INDEX_URL,
        "setup_completed": True,
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    log(f"✓ Configuration saved to {config_file}")


def main():
    """Main setup function."""
    log("Starting non-interactive setup for Moondream Station")
    log(f"Target CUDA version: {CUDA_VERSION}")

    try:
        # Step 1: Install PyTorch with CUDA support
        install_pytorch_and_deps()

        # Step 2: Install backend requirements and download backends
        install_backend_requirements()

        # Step 3: Pre-download moondream2 model
        download_moondream2_model()

        # Step 4: Create configuration
        create_config()

        log("=" * 60)
        log("✓ Setup completed successfully!")
        log("Moondream Station is ready to use with moondream2 model")
        log("=" * 60)

    except Exception as e:
        log("=" * 60)
        log(f"✗ Setup failed: {e}")
        log("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
