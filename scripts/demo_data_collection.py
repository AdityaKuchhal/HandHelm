#!/usr/bin/env python3
"""
Demo script for Hand Helm data collection.

This script demonstrates the data collection functionality
with a simple interactive demo.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_collection import DataCollector
from src.data_preprocessing import DataPreprocessor
from src.config import SUPPORTED_GESTURES

def demo_data_collection():
    """Demonstrate data collection functionality."""
    print("🎯 Hand Helm - Data Collection Demo")
    print("=" * 50)
    
    print("This demo will show you how to collect gesture data.")
    print(f"Supported gestures: {', '.join(SUPPORTED_GESTURES)}")
    print("\nNote: This demo will collect a small amount of data for testing.")
    
    # Create collector
    collector = DataCollector()
    
    try:
        # Initialize camera
        if not collector.initialize_camera():
            print("❌ Camera initialization failed. Please check your camera.")
            return
        
        print("\n📹 Camera initialized successfully!")
        print("You can now collect gesture data.")
        
        # Demo with a few gestures
        demo_gestures = ['fist', 'open_palm', 'thumbs_up']
        
        for gesture in demo_gestures:
            print(f"\n🎯 Collecting data for '{gesture}'")
            print("Press 'c' to capture samples, 'q' to quit, 's' to skip")
            
            # Collect just 5 samples for demo
            success = collector.collect_gesture_data(gesture, num_samples=5)
            
            if success:
                print(f"✅ Successfully collected data for '{gesture}'")
            else:
                print(f"⚠️ Skipped '{gesture}'")
            
            # Ask if user wants to continue
            if gesture != demo_gestures[-1]:  # Not the last gesture
                continue_collection = input(f"\nContinue to next gesture? (y/n): ").lower().strip()
                if continue_collection != 'y':
                    break
        
        # Save collected data
        if collector.collected_data:
            collector.save_collected_data("demo_gesture_data.csv")
            print("\n🎉 Demo data collection completed!")
            print("You can now run the preprocessing script to prepare the data for training.")
        else:
            print("\n⚠️ No data was collected during the demo.")
    
    except KeyboardInterrupt:
        print("\n🛑 Demo interrupted by user.")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
    finally:
        collector.cleanup()

def demo_data_preprocessing():
    """Demonstrate data preprocessing functionality."""
    print("\n🔧 Hand Helm - Data Preprocessing Demo")
    print("=" * 50)
    
    # Check if we have any raw data
    raw_files = [f for f in os.listdir('data/raw') if f.endswith('.csv')]
    
    if not raw_files:
        print("❌ No raw data files found.")
        print("Please run data collection first.")
        return
    
    print(f"Found {len(raw_files)} raw data file(s):")
    for i, file in enumerate(raw_files, 1):
        print(f"   {i}. {file}")
    
    # Use the most recent file
    latest_file = max(raw_files, key=lambda x: os.path.getmtime(os.path.join('data/raw', x)))
    print(f"\nUsing latest file: {latest_file}")
    
    try:
        preprocessor = DataPreprocessor()
        
        # Load and preprocess data
        print("\n📂 Loading raw data...")
        df = preprocessor.load_raw_data(os.path.join('data/raw', latest_file))
        
        print(f"📊 Loaded {len(df)} samples")
        print(f"Gestures: {df['gesture'].value_counts().to_dict()}")
        
        print("\n🔧 Preprocessing data...")
        X_train, X_val, y_train, y_val = preprocessor.preprocess_data(df)
        
        print("\n💾 Saving preprocessed data...")
        save_info = preprocessor.save_preprocessed_data(X_train, X_val, y_train, y_val)
        
        print("\n✅ Data preprocessing completed!")
        print("The data is now ready for model training.")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        import traceback
        traceback.print_exc()

def main():
    """Main demo function."""
    print("🚀 Welcome to Hand Helm Data Collection Demo!")
    print("=" * 60)
    
    while True:
        print("\nChoose a demo:")
        print("1. Data Collection Demo")
        print("2. Data Preprocessing Demo")
        print("3. Exit")
        
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == '1':
            demo_data_collection()
        elif choice == '2':
            demo_data_preprocessing()
        elif choice == '3':
            print("👋 Thanks for trying Hand Helm!")
            break
        else:
            print("❌ Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
