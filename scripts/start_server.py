#!/usr/bin/env python3
"""Command-line interface for Dreamspace Co-Pilot server."""

import argparse
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dreamspace.servers.api_server import run_server
from dreamspace.config.settings import Config


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Dreamspace Co-Pilot Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with Kandinsky backend
  %(prog)s --backend kandinsky21_server --port 8000
   
  # GPU selection examples
  %(prog)s --backend kandinsky21_server --gpus 0 --port 8001
  %(prog)s --backend kandinsky21_server --gpus 1 --port 8001
  %(prog)s --backend kandinsky21_server --gpus 0,1 --port 8001
  %(prog)s --backend kandinsky21_server --gpus auto --port 8001
  
  # SD 1.5 with safety checker disabled (fixes false positives)
  %(prog)s --backend sd15_server --disable-safety-checker --port 8001
        """,
    )

    parser.add_argument(
        "--backend",
        default="kandinsky21_server",
        choices=[
            "kandinsky22_server",
            "kandinsky21_server",
            "sd15_server",
            "sd21_server",
        ],
        help="Backend type to use (default: kandinsky21_server)",
    )

    parser.add_argument("--config", help="Path to configuration file (YAML or JSON)")

    parser.add_argument(
        "--host", default="localhost", help="Host to bind to (default: localhost)"
    )

    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind to (default: 8000)"
    )

    parser.add_argument(
        "--workers", type=int, default=1, help="Number of worker processes (default: 1)"
    )

    parser.add_argument(
        "--auth", action="store_true", help="Enable API key authentication"
    )

    parser.add_argument(
        "--api-key", help="API key for authentication (required if --auth is used)"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to run models on (overrides config)",
    )

    # GPU selection arguments
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="GPU selection: 'auto' (use all), '0' (first GPU), '1' (second GPU), '0,1' (both GPUs), or specific GPU IDs (default: auto)",
    )

    parser.add_argument(
        "--gpu-memory-fraction",
        type=float,
        default=0.9,
        help="Fraction of GPU memory to use (0.1-1.0, default: 0.9)",
    )

    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)",
    )

    # Safety checker arguments
    parser.add_argument(
        "--disable-safety-checker",
        action="store_true",
        help="Disable NSFW safety checker for SD 1.5 (fixes false positives)",
    )

    args = parser.parse_args()

    # Validate authentication arguments
    if args.auth and not args.api_key:
        parser.error("--api-key is required when --auth is enabled")

    # Load configuration if provided
    config = None
    if args.config:
        if not os.path.exists(args.config):
            print(f"❌ Configuration file not found: {args.config}")
            sys.exit(1)
        try:
            config = Config(args.config)
            print(f"✅ Loaded configuration from: {args.config}")
        except Exception as e:
            print(f"❌ Failed to load configuration: {e}")
            sys.exit(1)

    # Override device in config if specified
    if args.device and config:
        for model_config in config.models.values():
            model_config.device = args.device
        print(f"🔧 Device override: {args.device}")

    # Set GPU environment based on selection
    if args.gpus != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        print(f"🎯 GPU Selection: Using GPU(s) {args.gpus}")
    else:
        print("🎯 GPU Selection: Auto (using all available GPUs)")

    print(f"🚀 Starting Dreamspace Co-Pilot Server")
    print(f"   Backend: {args.backend}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Authentication: {'enabled' if args.auth else 'disabled'}")
    print(f"   Device: {args.device or 'from config'}")
    if args.gpus != "auto":
        print(
            f"   GPU Configuration: {args.gpus} (memory fraction: {args.gpu_memory_fraction})"
        )
    if args.disable_safety_checker:
        print(f"   Safety Checker: disabled (fixes false positives)")

    try:
        run_server(
            backend_type=args.backend,
            host=args.host,
            port=args.port,
            workers=args.workers,
            enable_auth=args.auth,
            api_key=args.api_key,
            gpus=args.gpus,
            disable_safety_checker=args.disable_safety_checker,
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
