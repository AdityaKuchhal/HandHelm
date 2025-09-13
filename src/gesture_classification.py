#!/usr/bin/env python3
"""
Gesture Classification Model for Hand Helm

This module contains the machine learning model for gesture classification,
including model architecture, training, evaluation, and inference capabilities.
"""

import os
import sys
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
import argparse

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import (
    MODEL_SAVE_PATH, SCALER_SAVE_PATH, LABEL_ENCODER_SAVE_PATH,
    BATCH_SIZE, NUM_EPOCHS, LEARNING_RATE, VALIDATION_SPLIT,
    EARLY_STOPPING_PATIENCE, GESTURE_CONFIDENCE_THRESHOLD
)

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
    from tensorflow.keras.utils import to_categorical
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False
    print("⚠️ TensorFlow not available. Using scikit-learn as fallback.")

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_preprocessing import DataPreprocessor

class GestureClassifier:
    """
    A comprehensive gesture classification system supporting both TensorFlow and scikit-learn models.
    
    This class provides:
    - Multiple model architectures (Neural Network, Random Forest, SVM)
    - Training and evaluation capabilities
    - Model persistence and loading
    - Real-time inference
    - Performance visualization
    """
    
    def __init__(self, model_type='neural_network'):
        """
        Initialize the gesture classifier.
        
        Args:
            model_type (str): Type of model to use ('neural_network', 'random_forest', 'svm')
        """
        self.model_type = model_type
        self.model = None
        self.scaler = None
        self.label_encoder = None
        self.class_names = None
        self.training_history = None
        self.is_trained = False
        
        # Create models directory if it doesn't exist
        os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    
    def create_neural_network(self, input_shape, num_classes):
        """
        Create a neural network model for gesture classification.
        
        Args:
            input_shape (tuple): Input shape (number of features,)
            num_classes (int): Number of gesture classes
        
        Returns:
            tf.keras.Model: Compiled neural network model
        """
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network models.")
        
        model = Sequential([
            # Input layer
            Dense(128, activation='relu', input_shape=input_shape),
            BatchNormalization(),
            Dropout(0.3),
            
            # Hidden layers
            Dense(256, activation='relu'),
            BatchNormalization(),
            Dropout(0.4),
            
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            Dropout(0.2),
            
            # Output layer
            Dense(num_classes, activation='softmax')
        ])
        
        # Compile model
        model.compile(
            optimizer=Adam(learning_rate=LEARNING_RATE),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def create_random_forest(self, n_estimators=100, max_depth=None, random_state=42):
        """
        Create a Random Forest classifier.
        
        Args:
            n_estimators (int): Number of trees in the forest
            max_depth (int): Maximum depth of the trees
            random_state (int): Random state for reproducibility
        
        Returns:
            RandomForestClassifier: Random Forest model
        """
        return RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def create_svm(self, kernel='rbf', C=1.0, gamma='scale'):
        """
        Create a Support Vector Machine classifier.
        
        Args:
            kernel (str): Kernel type ('rbf', 'linear', 'poly')
            C (float): Regularization parameter
            gamma (str or float): Kernel coefficient
        
        Returns:
            SVC: Support Vector Machine model
        """
        return SVC(
            kernel=kernel,
            C=C,
            gamma=gamma,
            probability=True,  # Enable probability estimates
            random_state=42
        )
    
    def load_preprocessed_data(self, timestamp=None):
        """
        Load preprocessed data for training.
        
        Args:
            timestamp (str): Timestamp of the data to load. If None, loads the most recent.
        
        Returns:
            dict: Loaded data and metadata
        """
        print("📂 Loading preprocessed data...")
        
        preprocessor = DataPreprocessor()
        data_info = preprocessor.load_preprocessed_data(timestamp)
        
        self.scaler = data_info['scaler']
        self.label_encoder = data_info['label_encoder']
        self.class_names = self.label_encoder.classes_
        
        print(f"✅ Loaded data with {len(self.class_names)} classes")
        print(f"   Classes: {', '.join(self.class_names)}")
        
        return data_info
    
    def train(self, X_train, y_train, X_val, y_val, class_names):
        """
        Train the gesture classification model.
        
        Args:
            X_train, y_train: Training data and labels
            X_val, y_val: Validation data and labels
            class_names: List of class names
        """
        print(f"🚀 Training {self.model_type} model...")
        print("=" * 50)
        
        self.class_names = class_names
        num_classes = len(class_names)
        
        if self.model_type == 'neural_network':
            self._train_neural_network(X_train, y_train, X_val, y_val, num_classes)
        elif self.model_type == 'random_forest':
            self._train_random_forest(X_train, y_train, X_val, y_val)
        elif self.model_type == 'svm':
            self._train_svm(X_train, y_train, X_val, y_val)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        self.is_trained = True
        print(f"✅ {self.model_type.title()} model training completed!")
    
    def _train_neural_network(self, X_train, y_train, X_val, y_val, num_classes):
        """Train neural network model."""
        if not TENSORFLOW_AVAILABLE:
            raise ImportError("TensorFlow is required for neural network training.")
        
        # Convert labels to categorical
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        
        # Create model
        self.model = self.create_neural_network((X_train.shape[1],), num_classes)
        
        # Print model summary
        print("\n📋 Model Architecture:")
        self.model.summary()
        
        # Define callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                MODEL_SAVE_PATH,
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]
        
        # Train model
        print(f"\n🏋️ Training for {NUM_EPOCHS} epochs...")
        self.training_history = self.model.fit(
            X_train, y_train_cat,
            validation_data=(X_val, y_val_cat),
            epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1
        )
        
        # Load best model
        self.model = load_model(MODEL_SAVE_PATH)
        print(f"💾 Best model saved to: {MODEL_SAVE_PATH}")
    
    def _train_random_forest(self, X_train, y_train, X_val, y_val):
        """Train Random Forest model."""
        print("🌲 Training Random Forest...")
        
        self.model = self.create_random_forest()
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_accuracy = self.model.score(X_val, y_val)
        print(f"✅ Random Forest training completed!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
    
    def _train_svm(self, X_train, y_train, X_val, y_val):
        """Train SVM model."""
        print("🔍 Training Support Vector Machine...")
        
        self.model = self.create_svm()
        self.model.fit(X_train, y_train)
        
        # Evaluate on validation set
        val_accuracy = self.model.score(X_val, y_val)
        print(f"✅ SVM training completed!")
        print(f"   Validation Accuracy: {val_accuracy:.4f}")
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the trained model.
        
        Args:
            X_test, y_test: Test data and labels
        
        Returns:
            dict: Evaluation metrics
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before evaluation.")
        
        print("📊 Evaluating model...")
        print("=" * 30)
        
        # Make predictions
        if self.model_type == 'neural_network':
            y_pred_proba = self.model.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis=1)
        else:
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"🎯 Test Accuracy: {accuracy:.4f}")
        print(f"📈 Test Samples: {len(y_test)}")
        
        # Classification report
        print("\n📋 Classification Report:")
        print(classification_report(y_test, y_pred, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        self._plot_confusion_matrix(cm, self.class_names)
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'confusion_matrix': cm
        }
    
    def _plot_confusion_matrix(self, cm, class_names):
        """Plot confusion matrix."""
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'confusion_matrix.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📊 Confusion matrix saved to: {plot_path}")
        plt.show()
    
    def plot_training_history(self):
        """Plot training history for neural network."""
        if self.model_type != 'neural_network' or self.training_history is None:
            print("⚠️ Training history only available for neural network models.")
            return
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot accuracy
        ax1.plot(self.training_history.history['accuracy'], label='Training Accuracy')
        ax1.plot(self.training_history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True)
        
        # Plot loss
        ax2.plot(self.training_history.history['loss'], label='Training Loss')
        ax2.plot(self.training_history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(os.path.dirname(MODEL_SAVE_PATH), 'training_history.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"📈 Training history saved to: {plot_path}")
        plt.show()
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Input features
        
        Returns:
            tuple: (predictions, probabilities)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        if self.model_type == 'neural_network':
            probabilities = self.model.predict(X)
            predictions = np.argmax(probabilities, axis=1)
        else:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
        
        return predictions, probabilities
    
    def _preprocess_landmarks(self, landmarks):
        """
        Preprocess raw landmarks to match training data format.
        
        Args:
            landmarks: Raw landmark data (42 features)
        
        Returns:
            np.ndarray: Preprocessed features (64 features)
        """
        # Reshape to 21 landmarks with x,y coordinates
        hand_landmarks = landmarks.reshape(-1, 2)
        
        # Normalize landmarks relative to wrist
        wrist = hand_landmarks[0]
        normalized_landmarks = hand_landmarks - wrist
        
        # Calculate distance features
        key_landmarks = [0, 4, 8, 12, 16, 20]  # Wrist, fingertips
        distances = []
        for i in range(len(key_landmarks)):
            for j in range(i + 1, len(key_landmarks)):
                p1 = hand_landmarks[key_landmarks[i]]
                p2 = hand_landmarks[key_landmarks[j]]
                distance = np.linalg.norm(p1 - p2)
                distances.append(distance)
        
        # Calculate angle features
        angle_points = [
            (0, 4, 8), (0, 8, 12), (0, 12, 16), (0, 16, 20),
            (4, 8, 12), (8, 12, 16), (12, 16, 20)
        ]
        
        angles = []
        for p1_idx, p2_idx, p3_idx in angle_points:
            p1 = hand_landmarks[p1_idx]
            p2 = hand_landmarks[p2_idx]
            p3 = hand_landmarks[p3_idx]
            
            v1 = p1 - p2
            v2 = p3 - p2
            
            if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
                angles.append(0)
                continue
            
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.arccos(cos_angle)
            angles.append(angle)
        
        # Combine all features
        features = np.concatenate([
            normalized_landmarks.flatten(),  # 42 features
            np.array(distances),             # ~15 features
            np.array(angles)                 # ~7 features
        ])
        
        return features.reshape(1, -1)
    
    def predict_gesture(self, landmarks, confidence_threshold=GESTURE_CONFIDENCE_THRESHOLD):
        """
        Predict gesture from hand landmarks with confidence threshold.
        
        Args:
            landmarks: Hand landmark data (raw 42 features or preprocessed 64 features)
            confidence_threshold: Minimum confidence for prediction
        
        Returns:
            dict: Prediction results
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions.")
        
        # Preprocess landmarks (assuming they're already in the right format)
        if len(landmarks.shape) == 1:
            landmarks = landmarks.reshape(1, -1)
        
        # Check if landmarks need preprocessing (42 features = raw, 64 features = preprocessed)
        if landmarks.shape[1] == 42:
            # Raw landmarks need preprocessing
            landmarks = self._preprocess_landmarks(landmarks)
        elif landmarks.shape[1] == 64:
            # Already preprocessed
            pass
        else:
            raise ValueError(f"Expected 42 or 64 features, got {landmarks.shape[1]}")
        
        # Scale features
        landmarks_scaled = self.scaler.transform(landmarks)
        
        # Make prediction
        predictions, probabilities = self.predict(landmarks_scaled)
        
        # Get prediction details
        predicted_class = self.class_names[predictions[0]]
        confidence = np.max(probabilities[0])
        
        # Apply confidence threshold
        if confidence < confidence_threshold:
            predicted_class = "unknown"
            confidence = 0.0
        
        return {
            'gesture': predicted_class,
            'confidence': confidence,
            'all_probabilities': dict(zip(self.class_names, probabilities[0]))
        }
    
    def save_model(self, filepath=None):
        """Save the trained model and associated components."""
        if not self.is_trained:
            raise ValueError("Model must be trained before saving.")
        
        if filepath is None:
            filepath = MODEL_SAVE_PATH
        
        print(f"💾 Saving model to: {filepath}")
        
        if self.model_type == 'neural_network':
            # Save TensorFlow model
            self.model.save(filepath)
        else:
            # Save scikit-learn model
            with open(filepath.replace('.h5', '.pkl'), 'wb') as f:
                pickle.dump(self.model, f)
        
        # Save scaler and label encoder
        with open(SCALER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        with open(LABEL_ENCODER_SAVE_PATH, 'wb') as f:
            pickle.dump(self.label_encoder, f)
        
        # Save metadata
        metadata = {
            'model_type': self.model_type,
            'class_names': self.class_names.tolist(),
            'num_classes': len(self.class_names),
            'trained_at': datetime.now().isoformat(),
            'tensorflow_available': TENSORFLOW_AVAILABLE
        }
        
        metadata_path = filepath.replace('.h5', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("✅ Model and components saved successfully!")
    
    def load_model(self, filepath=None):
        """Load a trained model and associated components."""
        if filepath is None:
            filepath = MODEL_SAVE_PATH
        
        print(f"📂 Loading model from: {filepath}")
        
        # Load metadata
        metadata_path = filepath.replace('.h5', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            self.model_type = metadata['model_type']
            self.class_names = np.array(metadata['class_names'])
        else:
            print("⚠️ No metadata found, assuming neural network model")
            self.model_type = 'neural_network'
        
        # Load model
        if self.model_type == 'neural_network' and TENSORFLOW_AVAILABLE:
            self.model = load_model(filepath)
        else:
            model_path = filepath.replace('.h5', '.pkl')
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
        
        # Load scaler and label encoder
        with open(SCALER_SAVE_PATH, 'rb') as f:
            self.scaler = pickle.load(f)
        
        with open(LABEL_ENCODER_SAVE_PATH, 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        self.is_trained = True
        print("✅ Model loaded successfully!")

def main():
    """Main function for training and evaluating gesture classification models."""
    parser = argparse.ArgumentParser(description="Hand Helm Gesture Classification")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate the model')
    parser.add_argument('--model-type', choices=['neural_network', 'random_forest', 'svm'],
                       default='neural_network', help='Type of model to use')
    parser.add_argument('--data-timestamp', help='Timestamp of preprocessed data to use')
    
    args = parser.parse_args()
    
    print("🎯 Hand Helm - Gesture Classification")
    print("=" * 50)
    
    # Initialize classifier
    classifier = GestureClassifier(model_type=args.model_type)
    
    if args.train:
        print(f"🚀 Training {args.model_type} model...")
        
        # Load preprocessed data
        data_info = classifier.load_preprocessed_data(args.data_timestamp)
        
        # Get data
        X_train = data_info['data']['X_train']
        X_val = data_info['data']['X_val']
        y_train = data_info['data']['y_train']
        y_val = data_info['data']['y_val']
        
        # Train model
        classifier.train(X_train, y_train, X_val, y_val, classifier.class_names)
        
        # Evaluate on validation set
        print("\n📊 Evaluating on validation set...")
        classifier.evaluate(X_val, y_val)
        
        # Plot training history for neural network
        if args.model_type == 'neural_network':
            classifier.plot_training_history()
        
        # Save model
        classifier.save_model()
        
        print("\n🎉 Training completed successfully!")
    
    elif args.evaluate:
        print("📊 Evaluating model...")
        
        # Load model
        classifier.load_model()
        
        # Load test data (using validation set for now)
        data_info = classifier.load_preprocessed_data(args.data_timestamp)
        X_val = data_info['data']['X_val']
        y_val = data_info['data']['y_val']
        
        # Evaluate
        results = classifier.evaluate(X_val, y_val)
        
        print(f"\n✅ Evaluation completed!")
        print(f"   Accuracy: {results['accuracy']:.4f}")
    
    else:
        print("❌ Please specify --train or --evaluate")
        print("Usage examples:")
        print("  python src/gesture_classification.py --train --model-type neural_network")
        print("  python src/gesture_classification.py --evaluate")

if __name__ == "__main__":
    main()
