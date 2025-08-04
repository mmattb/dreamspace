#!/usr/bin/env python3
"""Launcher for Dreamspace keyboard navigation.

Choose between different navigation modes and examples.
"""

import sys
import os
import subprocess

def main():
    print("🚀 Dreamspace Navigation Examples")
    print("=" * 40)
    print()
    print("Choose your navigation mode:")
    print("1. Animated Navigation (recommended) ⭐")
    print("2. Local model navigation (requires local GPU)")
    print("3. Simple remote navigation")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("🎬 Starting animated navigation (using main launcher)...")
            launcher_path = os.path.join(os.path.dirname(__file__), "..", "run_navigation.py")
            try:
                subprocess.run([sys.executable, launcher_path])
            except KeyboardInterrupt:
                print("\\n👋 Animated navigation stopped.")
            break
            
        elif choice == "2":
            print("🔮 Starting local keyboard navigation...")
            script_path = os.path.join(os.path.dirname(__file__), "keyboard_navigation.py")
            try:
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                print("\\n👋 Local navigation stopped.")
            break
            
        elif choice == "3":
            print("� Starting simple remote navigation...")
            script_path = os.path.join(os.path.dirname(__file__), "remote_keyboard_navigation.py")
            try:
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                print("\\n👋 Remote navigation stopped.")
            break
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
