# data_preprocessing.py
import cv2
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json

class DataPreprocessor:
    def __init__(self, data_dir="dataset", img_size=(224, 224)):
        self.data_dir = data_dir
        self.img_size = img_size
        self.classes = []
        self.data = []
        self.labels = []
    
    def load_and_preprocess_data(self):
        """Load and preprocess all images"""
        print("🔄 Loading and preprocessing data...")
        
        # Find all classes from directory structure
        self.classes = [d for d in os.listdir(self.data_dir) 
                       if os.path.isdir(os.path.join(self.data_dir, d))]
        self.classes.sort()
        
        print(f"Found classes: {self.classes}")
        
        for class_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_dir, class_name)
            image_files = [f for f in os.listdir(class_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            
            print(f"Processing {class_name}: {len(image_files)} images")
            
            for image_file in image_files:
                image_path = os.path.join(class_path, image_file)
                
                try:
                    # Load and preprocess image
                    image = cv2.imread(image_path)
                    if image is not None:
                        # Resize to consistent size
                        image = cv2.resize(image, self.img_size)
                        
                        # Normalize pixel values
                        image = image.astype(np.float32) / 255.0
                        
                        self.data.append(image)
                        self.labels.append(class_idx)
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
        
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        
        print(f"✅ Loaded {len(self.data)} images across {len(self.classes)} classes")
        return self.data, self.labels, self.classes
    
    def train_test_split(self, test_size=0.2, validation_size=0.2):
        """Split data into train, validation, and test sets"""
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            self.data, self.labels, test_size=test_size, random_state=42, stratify=self.labels
        )
        
        # Second split: separate validation set from remaining data
        val_ratio = validation_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        print(f"📊 Dataset split:")
        print(f"  Training: {len(X_train)} images")
        print(f"  Validation: {len(X_val)} images") 
        print(f"  Test: {len(X_test)} images")
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def save_dataset_info(self):
        """Save dataset information to JSON file"""
        dataset_info = {
            'num_classes': len(self.classes),
            'classes': self.classes,
            'class_distribution': {},
            'total_images': len(self.data),
            'image_size': self.img_size,
            'preprocessing': 'resize_224x224, normalize_0-1'
        }
        
        # Count images per class
        for class_idx, class_name in enumerate(self.classes):
            count = np.sum(self.labels == class_idx)
            dataset_info['class_distribution'][class_name] = int(count)
        
        # Save to file
        info_path = os.path.join(self.data_dir, 'dataset_info.json')
        with open(info_path, 'w') as f:
            json.dump(dataset_info, f, indent=2)
        
        print(f"💾 Dataset info saved to: {info_path}")
        return dataset_info

# Usage
if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    data, labels, classes = preprocessor.load_and_preprocess_data()
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = preprocessor.train_test_split()
    dataset_info = preprocessor.save_dataset_info()