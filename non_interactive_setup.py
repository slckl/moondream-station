#!/usr/bin/env python3
"""
Non-interactive setup script for Moondream Station in Docker environments.
This script sets up CUDA backend and downloads the moondream2 model for inference.
"""

import json
import subprocess
import sys
from pathlib import Path

LOCAL_MANIFEST_PATH = Path(__file__).parent / "moondream2.manifest.json"
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
        "current_model": "moondream-2",
        "models_dir": str(app_dir / "models"),
        "manifest_url": str(LOCAL_MANIFEST_PATH),
    }

    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    log(f"✓ Configuration saved to {config_file}")
    return config


def download_backend_and_model():
    """Download backend using ManifestManager and initialize model with specific revision."""
    log("Downloading backend and moondream2 model")

    # Import here after dependencies are installed
    from moondream_station.core.config import ConfigManager
    from moondream_station.core.manifest import ManifestManager

    # Create config manager
    config_manager = ConfigManager()

    # Create manifest manager
    manifest_manager = ManifestManager(config_manager)

    # Load manifest
    log(f"Loading local manifest from {LOCAL_MANIFEST_PATH}")
    success = manifest_manager.load_manifest(str(LOCAL_MANIFEST_PATH))
    if not success:
        raise Exception("Failed to load manifest")

    log("✓ Manifest loaded successfully")

    # Get the moondream-2 model info
    models = manifest_manager.get_models()
    if "moondream-2" not in models:
        raise Exception("moondream-2 model not found in manifest")

    model_info = models["moondream-2"]
    backend_id = model_info.backend

    log(f"Model: {model_info.name}")
    log(f"Backend: {backend_id}")

    # Download the backend
    log(f"Downloading backend: {backend_id}")
    if not manifest_manager.download_backend(backend_id):
        raise Exception(f"Failed to download backend: {backend_id}")

    log(f"✓ Backend {backend_id} downloaded and requirements installed")

    # Load the backend and initialize it with the specific revision
    log("Loading backend module")
    backend_module = manifest_manager.load_backend(backend_id)
    if not backend_module:
        raise Exception(f"Failed to load backend module: {backend_id}")

    log("✓ Backend module loaded")

    # Initialize backend with moondream2 revision 2025-06-21
    log("Initializing backend with moondream2 revision 2025-06-21")

    # Prepare args with the specific revision
    init_args = model_info.args.copy()
    init_args["revision_id"] = "2025-06-21"
    init_args["local_files_only"] = False  # Allow downloading during setup

    log(f"Backend init args: {init_args}")

    if hasattr(backend_module, "init_backend"):
        backend_module.init_backend(**init_args)
        log("✓ Backend initialized with args")

    # Trigger model download by calling get_model_service
    # This will download the model with the specified revision
    log("Triggering model download (this may take a while)...")
    if hasattr(backend_module, "get_model_service"):
        model_service = backend_module.get_model_service()
        log(f"✓ Model loaded successfully")
        log(f"Model device: {model_service.device}")
        log(f"Model revision: {model_service.revision}")
    else:
        log("⚠ Backend does not have get_model_service function")

    log("✓ Backend and model setup completed")


def main():
    """Main setup function."""
    log("Starting non-interactive setup for Moondream Station")
    log(f"Target CUDA version: {CUDA_VERSION}")

    try:
        # Step 1: Install PyTorch with CUDA support
        install_pytorch_and_deps()

        # Step 2: Create initial configuration
        create_config()

        # Step 3: Download backend and model using ManifestManager
        download_backend_and_model()

        log("=" * 60)
        log("✓ Setup completed successfully!")
        log("Moondream Station is ready to use with moondream2 model")
        log("Model revision: 2025-06-21")
        log("=" * 60)

    except Exception as e:
        log("=" * 60)
        log(f"✗ Setup failed: {e}")
        log("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
