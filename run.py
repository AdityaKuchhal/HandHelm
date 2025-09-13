#!/usr/bin/env python3
"""
Hand Helm - Quick Start Script

This script provides an easy way to run Hand Helm with different options.
"""

import os
import sys
import subprocess

def main():
    """Main function with menu options."""
    print("🎯 Hand Helm - Gesture Control System")
    print("=" * 50)
    print("Choose an option:")
    print("1. Run Trained Model (BEST!)")
    print("2. Run Improved Version (Rule-based)")
    print("3. Run Original Version")
    print("4. Test Gesture Detection Only")
    print("5. Test Computer Control")
    print("6. Collect Real Data")
    print("7. Full Training Pipeline")
    print("8. Exit")
    
    while True:
        try:
            choice = input("\nEnter your choice (1-8): ").strip()
            
            if choice == '1':
                print("\n🚀 Starting Trained Model Hand Helm...")
                subprocess.run([sys.executable, "src/app_trained.py"])
                break
            
            elif choice == '2':
                print("\n🚀 Starting Improved Hand Helm...")
                subprocess.run([sys.executable, "src/app_improved.py"])
                break
            
            elif choice == '3':
                print("\n🚀 Starting Original Hand Helm...")
                subprocess.run([sys.executable, "src/app.py"])
                break
            
            elif choice == '4':
                print("\n🧪 Testing Gesture Detection...")
                subprocess.run([sys.executable, "src/simple_gesture_detector.py"])
                break
            
            elif choice == '5':
                print("\n🎮 Testing Computer Control...")
                subprocess.run([sys.executable, "src/computer_control.py"])
                break
            
            elif choice == '6':
                print("\n📹 Starting Data Collection...")
                subprocess.run([sys.executable, "src/data_collection.py"])
                break
            
            elif choice == '7':
                print("\n🚀 Starting Full Training Pipeline...")
                subprocess.run([sys.executable, "scripts/full_training_pipeline.py"])
                break
            
            elif choice == '8':
                print("👋 Goodbye!")
                break
            
            else:
                print("❌ Invalid choice. Please enter 1-8.")
        
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
