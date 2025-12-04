# train_model.py
import tensorflow as tf
from tensorflow import keras
# Ensure data_preprocessing.py is in the same directory or update the import path accordingly
from dataprocesssing import DataPreprocessor
import os
import datetime

class ModelTrainer:
    def __init__(self):
        self.model = None
        self.history = None
    
    def create_model(self, num_classes, input_shape=(224, 224, 3)):
        """Create a simple CNN model for behavior classification"""
        model = keras.Sequential([
            # Feature extraction
            keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
            keras.layers.MaxPooling2D(2, 2),
            
            keras.layers.Conv2D(64, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            
            keras.layers.Conv2D(128, (3, 3), activation='relu'),
            keras.layers.MaxPooling2D(2, 2),
            
            # Classification
            keras.layers.Flatten(),
            keras.layers.Dense(512, activation='relu'),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, data_dir="dataset", epochs=20):
        """Train the model on collected data"""
        # Preprocess data
        preprocessor = DataPreprocessor(data_dir)
        data, labels, classes = preprocessor.load_and_preprocess_data()
        (X_train, y_train), (X_val, y_val), _ = preprocessor.train_test_split()
        
        # Create model
        self.model = self.create_model(num_classes=len(classes))
        
        print("🧠 Model Architecture:")
        self.model.summary()
        
        # Callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.2, patience=3)
        ]
        
        # Train model
        print("🚀 Starting training...")
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=32,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save model
        model_dir = "models"
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(model_dir, f"assault_detector_{timestamp}.h5")
        self.model.save(model_path)
        
        print(f"💾 Model saved to: {model_path}")
        return self.history
    
    def evaluate_model(self, test_data):
        """Evaluate model on test data"""
        if self.model and test_data:
            X_test, y_test = test_data
            test_loss, test_accuracy = self.model.evaluate(X_test, y_test, verbose=0)
            print(f"📊 Test Accuracy: {test_accuracy:.2%}")
            print(f"📊 Test Loss: {test_loss:.4f}")
            return test_accuracy

# Usage
if __name__ == "__main__":
    trainer = ModelTrainer()
    history = trainer.train(epochs=20)