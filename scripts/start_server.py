#!/usr/bin/env python3
"""Command-line interface for Dreamspace Co-Pilot server."""

import argparse
import sys
import os

# Add src to path for development
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

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
  %(prog)s --backend kandinsky_local --port 8000
  
  # Start server with authentication
  %(prog)s --backend sd_local --auth --api-key your-secret-key
  
  # Start production server
  %(prog)s --backend kandinsky_local --host 0.0.0.0 --port 8000 --workers 2
        """
    )
    
    parser.add_argument(
        "--backend", 
        default="kandinsky21_server",
        choices=[
            "kandinsky_local", "kandinsky21_server",
            "sd_local", "sd15_server", "sd21_server", 
            "remote"
        ],
        help="Backend type to use (default: kandinsky21_server)"
    )
    
    parser.add_argument(
        "--config",
        help="Path to configuration file (YAML or JSON)"
    )
    
    parser.add_argument(
        "--host", 
        default="localhost",
        help="Host to bind to (default: localhost)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8000,
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers", 
        type=int, 
        default=1,
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--auth", 
        action="store_true",
        help="Enable API key authentication"
    )
    
    parser.add_argument(
        "--api-key",
        help="API key for authentication (required if --auth is used)"
    )
    
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "mps"],
        help="Device to run models on (overrides config)"
    )
    
    parser.add_argument(
        "--log-level",
        default="info",
        choices=["debug", "info", "warning", "error"],
        help="Log level (default: info)"
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
    
    print(f"🚀 Starting Dreamspace Co-Pilot Server")
    print(f"   Backend: {args.backend}")
    print(f"   Host: {args.host}")
    print(f"   Port: {args.port}")
    print(f"   Workers: {args.workers}")
    print(f"   Authentication: {'enabled' if args.auth else 'disabled'}")
    print(f"   Device: {args.device or 'from config'}")
    
    try:
        run_server(
            backend_type=args.backend,
            host=args.host,
            port=args.port,
            workers=args.workers,
            enable_auth=args.auth,
            api_key=args.api_key
        )
    except KeyboardInterrupt:
        print("\n🛑 Server stopped by user")
    except Exception as e:
        print(f"❌ Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
