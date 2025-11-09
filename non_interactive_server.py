#!/usr/bin/env python3
"""
Non-interactive server script for Moondream Station.
Starts the REST API server without any interactive UI or REPL.

Usage:
    python non_interactive_server.py [--host HOST] [--port PORT] [--manifest URL]

    --host: Host to bind to (default: 0.0.0.0 for Docker compatibility)
    --port: Port to bind to (default: 2020)
    --manifest: Manifest URL or local path

Example:
    python non_interactive_server.py --host 0.0.0.0 --port 2020
"""

import signal
import sys
import time
from pathlib import Path

# Add moondream_station to path
sys.path.insert(0, str(Path(__file__).parent))

from moondream_station.core.config import SERVICE_HOST, SERVICE_PORT, ConfigManager
from moondream_station.core.manifest import ManifestManager
from moondream_station.core.rest_server import RestServer

DEFAULT_MANIFEST_URL = "https://m87-md-prod-assets.s3.us-west-2.amazonaws.com/station/mds2/production_manifest.json"
DEFAULT_HOST = "0.0.0.0"  # Bind to all interfaces for Docker compatibility


class NonInteractiveServer:
    """Non-interactive server runner for Moondream Station."""

    def __init__(
        self,
        manifest_url: str = DEFAULT_MANIFEST_URL,
        host: str = None,
        port: int = None,
    ):
        self.manifest_url = manifest_url
        self.host = host or DEFAULT_HOST
        self.port = port or SERVICE_PORT
        self.config = None
        self.manifest_manager = None
        self.rest_server = None
        self.running = False

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully."""
        print(f"\n[SERVER] Received signal {signum}, shutting down gracefully...")
        self.running = False

    def _log(self, message: str):
        """Print a log message."""
        print(f"[SERVER] {message}", flush=True)

    def _setup(self):
        """Set up configuration and manifest."""
        self._log("Initializing configuration...")
        self.config = ConfigManager()

        self._log(f"Loading manifest from {self.manifest_url}...")
        self.manifest_manager = ManifestManager(self.config)

        try:
            self.manifest_manager.load_manifest(self.manifest_url)
            self._log("✓ Manifest loaded successfully")
        except Exception as e:
            self._log(f"✗ Failed to load manifest: {e}")
            raise

        # Determine which model to use
        current_model = self.config.get("current_model")

        if not current_model:
            # Find default model or first available model
            manifest = self.manifest_manager.get_manifest()
            if manifest and manifest.models:
                # Look for default model
                default_model = None
                for model_id, model_info in manifest.models.items():
                    if model_info.is_default:
                        default_model = model_id
                        break

                # Use default or first model
                current_model = default_model or list(manifest.models.keys())[0]
                self.config.set("current_model", current_model)
                self._log(f"Set current model to: {current_model}")
            else:
                raise RuntimeError("No models available in manifest")
        else:
            self._log(f"Using configured model: {current_model}")

        return current_model

    def _start_server(self, model_name: str):
        """Start the REST server."""
        self._log(f"Starting REST server on {self.host}:{self.port}...")
        self._log(f"Initializing model: {model_name}")

        # Create and start the REST server
        self.rest_server = RestServer(
            self.config, self.manifest_manager, session_state=None, analytics=None
        )

        if self.rest_server.start(host=self.host, port=self.port):
            self._log("✓ Server started successfully")
            self._log(f"✓ Model '{model_name}' loaded and ready")
            self._log(f"✓ Server listening at http://{self.host}:{self.port}")
            self._log(f"✓ Health check: http://{self.host}:{self.port}/health")
            self._log(f"✓ API endpoints: http://{self.host}:{self.port}/v1/...")
            return True
        else:
            self._log("✗ Failed to start server")
            self._log(f"✗ Possible causes:")
            self._log(f"  - Model '{model_name}' backend not initialized")
            self._log(f"  - Port {self.port} already in use")
            self._log(f"  - Insufficient GPU memory")
            return False

    def run(self):
        """Run the server until interrupted."""
        try:
            # Setup configuration and manifest
            model_name = self._setup()

            # Start the server
            if not self._start_server(model_name):
                sys.exit(1)

            # Mark as running
            self.running = True

            self._log("=" * 60)
            self._log("Server is running. Press Ctrl+C to stop.")
            self._log("=" * 60)

            # Keep the main thread alive
            while self.running and self.rest_server.is_running():
                time.sleep(1)

        except KeyboardInterrupt:
            self._log("Received keyboard interrupt")
        except Exception as e:
            self._log(f"Error: {e}")
            import traceback

            traceback.print_exc()
            sys.exit(1)
        finally:
            self._shutdown()

    def _shutdown(self):
        """Shutdown the server gracefully."""
        self._log("Shutting down server...")

        if self.rest_server:
            try:
                self.rest_server.stop()
                self._log("✓ Server stopped")
            except Exception as e:
                self._log(f"Error stopping server: {e}")

        self._log("Shutdown complete")


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Moondream Station Non-Interactive Server"
    )
    parser.add_argument(
        "--host",
        type=str,
        default=None,
        help=f"Host to bind to (default: {DEFAULT_HOST})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help=f"Port to bind to (default: {SERVICE_PORT})",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=DEFAULT_MANIFEST_URL,
        help="Manifest URL or local path",
    )

    args = parser.parse_args()

    # Create and run the server
    server = NonInteractiveServer(
        manifest_url=args.manifest, host=args.host, port=args.port
    )
    server.run()


if __name__ == "__main__":
    main()
