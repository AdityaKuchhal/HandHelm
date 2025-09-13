#!/usr/bin/env python3
"""
Data Management Utilities for Hand Helm

This script provides utilities for managing gesture training data:
- View data statistics
- Merge multiple datasets
- Clean and validate data
- Export data in different formats
"""

import os
import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, SUPPORTED_GESTURES

class DataManager:
    """Utility class for managing gesture training data."""
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
    
    def list_datasets(self):
        """List all available datasets."""
        print("📁 Available Raw Datasets:")
        print("-" * 40)
        
        raw_files = [f for f in os.listdir(self.raw_data_dir) if f.endswith('.csv')]
        if not raw_files:
            print("   No raw datasets found.")
        else:
            for i, file in enumerate(raw_files, 1):
                filepath = os.path.join(self.raw_data_dir, file)
                size = os.path.getsize(filepath)
                mod_time = datetime.fromtimestamp(os.path.getmtime(filepath))
                print(f"   {i}. {file}")
                print(f"      Size: {size:,} bytes")
                print(f"      Modified: {mod_time.strftime('%Y-%m-%d %H:%M:%S')}")
                print()
        
        print("📁 Available Processed Datasets:")
        print("-" * 40)
        
        processed_files = [f for f in os.listdir(self.processed_data_dir) if f.endswith('.npy')]
        if not processed_files:
            print("   No processed datasets found.")
        else:
            # Group by timestamp
            timestamps = set()
            for file in processed_files:
                if '_metadata.json' not in file:
                    timestamp = file.split('_')[-1].replace('.npy', '')
                    timestamps.add(timestamp)
            
            for i, timestamp in enumerate(sorted(timestamps, reverse=True), 1):
                print(f"   {i}. Processed data from {timestamp}")
                print(f"      Files: {len([f for f in processed_files if timestamp in f])}")
                print()
    
    def analyze_dataset(self, filepath):
        """Analyze a dataset and show statistics."""
        print(f"📊 Analyzing dataset: {os.path.basename(filepath)}")
        print("=" * 50)
        
        try:
            df = pd.read_csv(filepath)
            
            # Basic info
            print(f"Total samples: {len(df)}")
            print(f"Features: {len([col for col in df.columns if col.startswith('landmark_')])}")
            
            # Gesture distribution
            print("\nGesture distribution:")
            gesture_counts = df['gesture'].value_counts()
            for gesture, count in gesture_counts.items():
                percentage = (count / len(df)) * 100
                print(f"   {gesture}: {count} samples ({percentage:.1f}%)")
            
            # Data quality checks
            print("\nData quality checks:")
            
            # Check for missing values
            missing_values = df.isnull().sum().sum()
            print(f"   Missing values: {missing_values}")
            
            # Check for duplicate samples
            duplicates = df.duplicated().sum()
            print(f"   Duplicate samples: {duplicates}")
            
            # Check landmark data quality
            landmark_cols = [col for col in df.columns if col.startswith('landmark_')]
            if landmark_cols:
                landmark_data = df[landmark_cols]
                
                # Check for extreme values (likely errors)
                extreme_values = ((landmark_data < 0) | (landmark_data > 1000)).sum().sum()
                print(f"   Extreme landmark values: {extreme_values}")
                
                # Check for zero values (likely missing data)
                zero_values = (landmark_data == 0).sum().sum()
                print(f"   Zero landmark values: {zero_values}")
            
            print(f"\n✅ Dataset analysis complete!")
            
        except Exception as e:
            print(f"❌ Error analyzing dataset: {e}")
    
    def merge_datasets(self, filepaths, output_filename=None):
        """Merge multiple datasets into one."""
        print("🔄 Merging datasets...")
        
        if not filepaths:
            print("❌ No files provided for merging.")
            return False
        
        merged_data = []
        
        for filepath in filepaths:
            try:
                df = pd.read_csv(filepath)
                merged_data.append(df)
                print(f"   ✅ Loaded {len(df)} samples from {os.path.basename(filepath)}")
            except Exception as e:
                print(f"   ❌ Error loading {filepath}: {e}")
                continue
        
        if not merged_data:
            print("❌ No valid datasets to merge.")
            return False
        
        # Combine all datasets
        combined_df = pd.concat(merged_data, ignore_index=True)
        
        # Remove duplicates
        original_count = len(combined_df)
        combined_df = combined_df.drop_duplicates()
        removed_duplicates = original_count - len(combined_df)
        
        if removed_duplicates > 0:
            print(f"   🧹 Removed {removed_duplicates} duplicate samples")
        
        # Generate output filename
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"merged_gesture_data_{timestamp}.csv"
        
        output_path = os.path.join(self.raw_data_dir, output_filename)
        
        # Save merged dataset
        combined_df.to_csv(output_path, index=False)
        
        print(f"✅ Merged dataset saved: {output_filename}")
        print(f"   Total samples: {len(combined_df)}")
        
        # Show final distribution
        print("\nFinal gesture distribution:")
        gesture_counts = combined_df['gesture'].value_counts()
        for gesture, count in gesture_counts.items():
            percentage = (count / len(combined_df)) * 100
            print(f"   {gesture}: {count} samples ({percentage:.1f}%)")
        
        return True
    
    def clean_dataset(self, filepath, output_filename=None):
        """Clean a dataset by removing invalid samples."""
        print(f"🧹 Cleaning dataset: {os.path.basename(filepath)}")
        
        try:
            df = pd.read_csv(filepath)
            original_count = len(df)
            
            # Remove rows with missing values
            df_clean = df.dropna()
            removed_missing = original_count - len(df_clean)
            
            # Remove duplicate samples
            df_clean = df_clean.drop_duplicates()
            removed_duplicates = len(df) - len(df_clean)
            
            # Remove samples with extreme landmark values
            landmark_cols = [col for col in df_clean.columns if col.startswith('landmark_')]
            if landmark_cols:
                # Keep only samples where all landmarks are within reasonable bounds
                valid_mask = ((df_clean[landmark_cols] >= 0) & (df_clean[landmark_cols] <= 1000)).all(axis=1)
                df_clean = df_clean[valid_mask]
                removed_extreme = len(df) - removed_duplicates - len(df_clean)
            else:
                removed_extreme = 0
            
            # Generate output filename
            if output_filename is None:
                base_name = os.path.splitext(os.path.basename(filepath))[0]
                output_filename = f"{base_name}_cleaned.csv"
            
            output_path = os.path.join(self.raw_data_dir, output_filename)
            df_clean.to_csv(output_path, index=False)
            
            print(f"✅ Dataset cleaned and saved: {output_filename}")
            print(f"   Original samples: {original_count}")
            print(f"   Cleaned samples: {len(df_clean)}")
            print(f"   Removed missing: {removed_missing}")
            print(f"   Removed duplicates: {removed_duplicates}")
            print(f"   Removed extreme values: {removed_extreme}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error cleaning dataset: {e}")
            return False
    
    def export_for_training(self, filepath, output_dir=None):
        """Export dataset in a format ready for training."""
        print(f"📤 Exporting dataset for training: {os.path.basename(filepath)}")
        
        try:
            df = pd.read_csv(filepath)
            
            if output_dir is None:
                output_dir = self.processed_data_dir
            
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)
            
            # Export features and labels separately
            landmark_cols = [col for col in df.columns if col.startswith('landmark_')]
            feature_data = df[landmark_cols].values
            labels = df['gesture'].values
            
            # Save as numpy arrays
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            features_path = os.path.join(output_dir, f"features_{timestamp}.npy")
            labels_path = os.path.join(output_dir, f"labels_{timestamp}.npy")
            
            np.save(features_path, feature_data)
            np.save(labels_path, labels)
            
            # Save metadata
            metadata = {
                'num_samples': len(df),
                'num_features': len(landmark_cols),
                'gestures': df['gesture'].unique().tolist(),
                'gesture_counts': df['gesture'].value_counts().to_dict(),
                'export_timestamp': timestamp
            }
            
            metadata_path = os.path.join(output_dir, f"metadata_{timestamp}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"✅ Training data exported:")
            print(f"   Features: {features_path}")
            print(f"   Labels: {labels_path}")
            print(f"   Metadata: {metadata_path}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error exporting dataset: {e}")
            return False

def main():
    """Main function for data management CLI."""
    parser = argparse.ArgumentParser(description="Hand Helm Data Management Tool")
    parser.add_argument('command', choices=['list', 'analyze', 'merge', 'clean', 'export'],
                       help='Command to execute')
    parser.add_argument('--files', nargs='+', help='Input files for the command')
    parser.add_argument('--output', help='Output filename')
    parser.add_argument('--output-dir', help='Output directory')
    
    args = parser.parse_args()
    
    manager = DataManager()
    
    if args.command == 'list':
        manager.list_datasets()
    
    elif args.command == 'analyze':
        if not args.files:
            print("❌ Please provide a file to analyze.")
            return
        manager.analyze_dataset(args.files[0])
    
    elif args.command == 'merge':
        if not args.files or len(args.files) < 2:
            print("❌ Please provide at least 2 files to merge.")
            return
        manager.merge_datasets(args.files, args.output)
    
    elif args.command == 'clean':
        if not args.files:
            print("❌ Please provide a file to clean.")
            return
        manager.clean_dataset(args.files[0], args.output)
    
    elif args.command == 'export':
        if not args.files:
            print("❌ Please provide a file to export.")
            return
        manager.export_for_training(args.files[0], args.output_dir)

if __name__ == "__main__":
    main()
