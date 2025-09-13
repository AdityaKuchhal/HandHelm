import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import sys

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    SCALER_SAVE_PATH, LABEL_ENCODER_SAVE_PATH,
    VALIDATION_SPLIT, SUPPORTED_GESTURES
)

class DataPreprocessor:
    """
    A class to preprocess hand gesture data for machine learning.
    
    This class handles:
    - Loading raw gesture data
    - Feature extraction and normalization
    - Train/validation split
    - Saving preprocessed data and encoders
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = None
        self.processed_data = None
        
        # Create processed data directory if it doesn't exist
        os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    def load_raw_data(self, filepath=None):
        """
        Load raw gesture data from CSV file.
        
        Args:
            filepath (str): Path to the CSV file. If None, loads the most recent file.
        
        Returns:
            pd.DataFrame: Loaded raw data
        """
        if filepath is None:
            # Find the most recent CSV file in raw data directory
            csv_files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.csv')]
            if not csv_files:
                raise FileNotFoundError("No CSV files found in raw data directory.")
            
            # Sort by modification time and get the most recent
            csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(RAW_DATA_DIR, x)), reverse=True)
            filepath = os.path.join(RAW_DATA_DIR, csv_files[0])
        
        print(f"📂 Loading data from: {filepath}")
        df = pd.read_csv(filepath)
        print(f"📊 Loaded {len(df)} samples")
        
        return df
    
    def extract_features(self, df):
        """
        Extract features from raw landmark data.
        
        Args:
            df (pd.DataFrame): Raw data with landmark coordinates
        
        Returns:
            np.ndarray: Extracted features
        """
        print("🔧 Extracting features...")
        
        # Get landmark columns (42 features: 21 landmarks x 2 coordinates)
        landmark_cols = [col for col in df.columns if col.startswith('landmark_')]
        
        if not landmark_cols:
            raise ValueError("No landmark columns found in the data.")
        
        # Extract landmark coordinates
        landmarks = df[landmark_cols].values
        
        # Calculate additional features
        features = []
        
        for i in range(len(landmarks)):
            # Reshape to 21 landmarks with x,y coordinates
            hand_landmarks = landmarks[i].reshape(-1, 2)
            
            # Basic features: normalized coordinates
            normalized_landmarks = self._normalize_landmarks(hand_landmarks)
            
            # Distance features: distances between key landmarks
            distance_features = self._calculate_distances(hand_landmarks)
            
            # Angle features: angles between key landmarks
            angle_features = self._calculate_angles(hand_landmarks)
            
            # Combine all features
            sample_features = np.concatenate([
                normalized_landmarks.flatten(),  # 42 features
                distance_features,               # ~20 features
                angle_features                   # ~15 features
            ])
            
            features.append(sample_features)
        
        features = np.array(features)
        print(f"✅ Extracted {features.shape[1]} features per sample")
        
        return features
    
    def _normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to the wrist (landmark 0).
        
        Args:
            landmarks (np.ndarray): Hand landmarks with shape (21, 2)
        
        Returns:
            np.ndarray: Normalized landmarks
        """
        wrist = landmarks[0]  # Wrist is landmark 0
        normalized = landmarks - wrist
        return normalized
    
    def _calculate_distances(self, landmarks):
        """
        Calculate distances between key landmarks.
        
        Args:
            landmarks (np.ndarray): Hand landmarks with shape (21, 2)
        
        Returns:
            np.ndarray: Distance features
        """
        # Key landmark indices for distance calculations
        key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist, fingertips
        
        distances = []
        for i in range(len(key_landmarks)):
            for j in range(i + 1, len(key_landmarks)):
                p1 = landmarks[key_landmarks[i]]
                p2 = landmarks[key_landmarks[j]]
                distance = np.linalg.norm(p1 - p2)
                distances.append(distance)
        
        return np.array(distances)
    
    def _calculate_angles(self, landmarks):
        """
        Calculate angles between key landmarks.
        
        Args:
            landmarks (np.ndarray): Hand landmarks with shape (21, 2)
        
        Returns:
            np.ndarray: Angle features
        """
        # Key landmark indices for angle calculations
        angle_points = [
            (0, 4, 8),   # Wrist to thumb tip to index tip
            (0, 8, 12),  # Wrist to index tip to middle tip
            (0, 12, 16), # Wrist to middle tip to ring tip
            (0, 16, 20), # Wrist to ring tip to pinky tip
            (4, 8, 12),  # Thumb tip to index tip to middle tip
            (8, 12, 16), # Index tip to middle tip to ring tip
            (12, 16, 20) # Middle tip to ring tip to pinky tip
        ]
        
        angles = []
        for p1_idx, p2_idx, p3_idx in angle_points:
            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            p3 = landmarks[p3_idx]
            
            # Calculate angle at p2
            v1 = p1 - p2
            v2 = p3 - p2
            
            # Avoid division by zero
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                angles.append(0)
                continue
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)  # Avoid numerical errors
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        return np.array(angles)
    
    def preprocess_data(self, df):
        """
        Complete preprocessing pipeline.
        
        Args:
            df (pd.DataFrame): Raw data
        
        Returns:
            tuple: (X_train, X_val, y_train, y_val, feature_columns)
        """
        print("🚀 Starting data preprocessing...")
        
        # Extract features
        X = self.extract_features(df)
        
        # Get labels
        y = df['gesture'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_encoded, 
            test_size=VALIDATION_SPLIT, 
            random_state=42, 
            stratify=y_encoded
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # Store feature column names
        self.feature_columns = [f'feature_{i}' for i in range(X.shape[1])]
        
        print(f"✅ Preprocessing complete!")
        print(f"   Training samples: {len(X_train_scaled)}")
        print(f"   Validation samples: {len(X_val_scaled)}")
        print(f"   Features per sample: {X_train_scaled.shape[1]}")
        print(f"   Classes: {len(self.label_encoder.classes_)}")
        
        return X_train_scaled, X_val_scaled, y_train, y_val
    
    def save_preprocessed_data(self, X_train, X_val, y_train, y_val, filename_prefix="processed_data"):
        """
        Save preprocessed data and encoders.
        
        Args:
            X_train, X_val, y_train, y_val: Preprocessed data
            filename_prefix (str): Prefix for saved files
        """
        timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        
        # Save preprocessed data
        data_files = {
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val
        }
        
        for name, data in data_files.items():
            filepath = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_{name}_{timestamp}.npy")
            np.save(filepath, data)
            print(f"💾 Saved {name}: {filepath}")
        
        # Save scaler
        scaler_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_scaler_{timestamp}.pkl")
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"💾 Saved scaler: {scaler_path}")
        
        # Save label encoder
        encoder_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_label_encoder_{timestamp}.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        print(f"💾 Saved label encoder: {encoder_path}")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'num_features': X_train.shape[1],
            'num_classes': len(self.label_encoder.classes_),
            'class_names': self.label_encoder.classes_.tolist(),
            'timestamp': timestamp
        }
        
        metadata_path = os.path.join(PROCESSED_DATA_DIR, f"{filename_prefix}_metadata_{timestamp}.json")
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"💾 Saved metadata: {metadata_path}")
        
        return {
            'data_files': data_files,
            'scaler_path': scaler_path,
            'encoder_path': encoder_path,
            'metadata_path': metadata_path
        }
    
    def load_preprocessed_data(self, timestamp=None):
        """
        Load preprocessed data and encoders.
        
        Args:
            timestamp (str): Timestamp of the data to load. If None, loads the most recent.
        
        Returns:
            dict: Loaded data and paths
        """
        if timestamp is None:
            # Find the most recent metadata file
            metadata_files = [f for f in os.listdir(PROCESSED_DATA_DIR) if f.endswith('_metadata.json')]
            if not metadata_files:
                raise FileNotFoundError("No preprocessed data found.")
            
            # Extract timestamp from filename
            timestamp = metadata_files[0].split('_')[-1].replace('.json', '')
        
        # Load data
        data = {}
        for split in ['X_train', 'X_val', 'y_train', 'y_val']:
            filepath = os.path.join(PROCESSED_DATA_DIR, f"processed_data_{split}_{timestamp}.npy")
            data[split] = np.load(filepath)
        
        # Load scaler and encoder
        scaler_path = os.path.join(PROCESSED_DATA_DIR, f"processed_data_scaler_{timestamp}.pkl")
        encoder_path = os.path.join(PROCESSED_DATA_DIR, f"processed_data_label_encoder_{timestamp}.pkl")
        
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        
        with open(encoder_path, 'rb') as f:
            label_encoder = pickle.load(f)
        
        return {
            'data': data,
            'scaler': scaler,
            'label_encoder': label_encoder,
            'timestamp': timestamp
        }

def main():
    """Main function to run data preprocessing."""
    print("🔧 Hand Helm - Data Preprocessing Tool")
    print("=" * 40)
    
    preprocessor = DataPreprocessor()
    
    try:
        # Load raw data
        df = preprocessor.load_raw_data()
        
        # Show data info
        print(f"\n📊 Data Overview:")
        print(f"   Total samples: {len(df)}")
        print(f"   Gestures: {df['gesture'].value_counts().to_dict()}")
        
        # Preprocess data
        X_train, X_val, y_train, y_val = preprocessor.preprocess_data(df)
        
        # Save preprocessed data
        save_info = preprocessor.save_preprocessed_data(X_train, X_val, y_train, y_val)
        
        print(f"\n🎉 Preprocessing completed successfully!")
        print(f"📁 All files saved in: {PROCESSED_DATA_DIR}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
