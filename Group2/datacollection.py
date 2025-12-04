# data_collection.py
import cv2
import os
import time
from datetime import datetime

class DataCollector:
    def __init__(self, data_dir="dataset"):
        self.data_dir = data_dir
        self.classes = [
            "normal_standing",
            "normal_walking", 
            "normal_sitting",
            "aggressive_pose",
            "fighting_stance",
            "raised_arms",
            "rapid_movement",
            "pushing_motion",
            "grabbing_motion",
            "falling_down"
        ]
        self.setup_directories()
    
    def setup_directories(self):
        """Create directory structure for data collection"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if not os.path.exists(class_path):
                os.makedirs(class_path)
                print(f"Created directory: {class_path}")
    
    def collect_images(self, class_name, num_images=5):
        """Collect images for a specific class using webcam"""
        class_path = os.path.join(self.data_dir, class_name)
        
        # Count existing images
        existing_images = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
        
        cap = cv2.VideoCapture(0)
        print(f"\n📸 Collecting {num_images} images for: {class_name}")
        print("Press SPACE to capture image, 'q' to quit early")
        
        count = existing_images
        try:
            while count < existing_images + num_images:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Display instructions on frame
                cv2.putText(frame, f"Class: {class_name}", (20, 40), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(frame, f"Images: {count}/{existing_images + num_images}", (20, 80), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, "SPACE: Capture  Q: Quit", (20, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                cv2.imshow('Data Collection', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord(' '):  # Space bar to capture
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    filename = f"{class_name}_{timestamp}.jpg"
                    filepath = os.path.join(class_path, filename)
                    
                    # Save image
                    cv2.imwrite(filepath, frame)
                    print(f"✅ Saved: {filename}")
                    count += 1
                    
                    # Show confirmation
                    cv2.putText(frame, "CAPTURED!", (200, 200), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.imshow('Data Collection', frame)
                    cv2.waitKey(500)  # Brief pause
                    
                elif key == ord('q'):
                    break
                    
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"🎉 Collected {count - existing_images} new images for {class_name}")
    
    def collect_all_classes(self, images_per_class=5):
        """Collect data for all classes sequentially"""
        print("🚀 Starting data collection for all classes...")
        print("=" * 50)
        
        for i, class_name in enumerate(self.classes):
            print(f"\n[{i+1}/{len(self.classes)}] Next: {class_name}")
            input("Press Enter when ready to collect this class...")
            self.collect_images(class_name, images_per_class)
        
        self.show_dataset_summary()
    
    def show_dataset_summary(self):
        """Show summary of collected dataset"""
        print("\n" + "=" * 50)
        print("📊 DATASET SUMMARY")
        print("=" * 50)
        
        total_images = 0
        for class_name in self.classes:
            class_path = os.path.join(self.data_dir, class_name)
            if os.path.exists(class_path):
                class_images = len([f for f in os.listdir(class_path) if f.endswith('.jpg')])
                total_images += class_images
                print(f"{class_name}: {class_images} images")
        
        print(f"\n📁 Total images: {total_images}")
        print(f"📁 Total classes: {len(self.classes)}")
        print(f"📁 Location: {os.path.abspath(self.data_dir)}")

# Usage
if __name__ == "__main__":
    collector = DataCollector()
    collector.collect_all_classes(images_per_class=5)
    