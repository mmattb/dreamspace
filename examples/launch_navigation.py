#!/usr/bin/env python3
"""Launcher for Dreamspace keyboard navigation.

Choose between local model or remote server.
"""

import sys
import os
import subprocess

def main():
    print("🚀 Dreamspace Keyboard Navigation Launcher")
    print("=" * 45)
    print()
    print("Choose your navigation mode:")
    print("1. Local model (requires local GPU/CPU inference)")
    print("2. Remote server (single images)")
    print("3. Remote server (animated batches) ⭐ NEW!")
    print("4. Exit")
    print()
    
    while True:
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == "1":
            print("🔮 Starting local keyboard navigation...")
            script_path = os.path.join(os.path.dirname(__file__), "keyboard_navigation.py")
            try:
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                print("\\n👋 Local navigation stopped.")
            break
            
        elif choice == "2":
            print("🌐 Starting remote keyboard navigation...")
            script_path = os.path.join(os.path.dirname(__file__), "remote_keyboard_navigation.py")
            try:
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                print("\\n👋 Remote navigation stopped.")
            break
            
        elif choice == "3":
            print("🎬 Starting animated remote navigation...")
            script_path = os.path.join(os.path.dirname(__file__), "animated_navigation.py")
            try:
                subprocess.run([sys.executable, script_path])
            except KeyboardInterrupt:
                print("\\n👋 Animated navigation stopped.")
            break
            
        elif choice == "4":
            print("👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()
