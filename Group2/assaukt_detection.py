# assault_detection.py
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import time
from collections import deque
import os

class AssaultDetectionSystem:
    def __init__(self, model_path=None):
        # Load trained model
        if model_path and os.path.exists(model_path):
            self.model = keras.models.load_model(model_path)
            print(f"✅ Loaded trained model: {model_path}")
        else:
            self.model = None
            print("⚠️  No trained model found. Using rule-based detection only.")
        
        # Detection parameters
        self.assault_threshold = 0.7
        self.confidence_threshold = 0.6
        self.alert_cooldown = 5  # seconds
        self.last_alert_time = 0
        
        # Tracking and history
        self.detection_history = deque(maxlen=10)
        self.alerts_log = []
        self.prediction_history = deque(maxlen=15)
        
        # Motion tracking
        self.prev_frame = None
        self.motion_history = deque(maxlen=10)
        
        # Class labels (should match your training data)
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
        
        # Violence-related classes (for assault detection)
        self.violence_classes = [
            "aggressive_pose",
            "fighting_stance", 
            "raised_arms",
            "pushing_motion",
            "grabbing_motion"
        ]
        
        print("🚨 Assault Detection System Initialized")
        print("📊 Using combined ML + rule-based detection")
        print("-" * 50)
    
    def preprocess_frame(self, frame):
        """Preprocess frame for model prediction"""
        # Resize to match training size
        processed = cv2.resize(frame, (224, 224))
        # Normalize pixel values
        processed = processed.astype(np.float32) / 255.0
        # Add batch dimension
        processed = np.expand_dims(processed, axis=0)
        return processed
    
    def predict_behavior(self, frame):
        """Use ML model to predict behavior class"""
        if self.model is None:
            return None, 0
        
        try:
            processed_frame = self.preprocess_frame(frame)
            predictions = self.model.predict(processed_frame, verbose=0)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            predicted_class = self.classes[predicted_class_idx]
            return predicted_class, confidence
        except Exception as e:
            print(f"❌ Prediction error: {e}")
            return None, 0
    
    def rule_based_detection(self, frame):
        """Fallback rule-based detection when no model available"""
        aggression_score = 0
        detected_objects = []
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)
        
        # Initialize previous frame
        if self.prev_frame is None:
            self.prev_frame = gray
            return aggression_score, detected_objects
        
        # Motion detection
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=2)
        
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:  # Large movements only
                x, y, w, h = cv2.boundingRect(contour)
                
                # Aspect ratio analysis
                aspect_ratio = w / h if h > 0 else 0
                
                # Aggressive pose: wide stance or raised arms
                if aspect_ratio > 1.8:  # Very wide = aggressive stance
                    aggression_score += 40
                    detected_objects.append((x, y, w, h, "AGGRESSIVE_STANCE", (0, 165, 255)))
                elif aspect_ratio < 0.4:  # Very tall = raised arms
                    aggression_score += 35
                    detected_objects.append((x, y, w, h, "RAISED_ARMS", (0, 0, 255)))
                else:
                    detected_objects.append((x, y, w, h, "PERSON", (0, 255, 0)))
                
                # Track movement speed
                self.motion_history.append((x, y, w, h))
                if len(self.motion_history) >= 3:
                    recent_motions = list(self.motion_history)[-3:]
                    x_movements = [pos[0] for pos in recent_motions]
                    if len(x_movements) >= 2:
                        movement_speed = abs(x_movements[-1] - x_movements[0])
                        if movement_speed > 50:  # Rapid horizontal movement
                            aggression_score += 25
                            detected_objects.append((x, y, w, h, "RAPID_MOVEMENT", (0, 255, 255)))
        
        self.prev_frame = gray
        return min(aggression_score, 100), detected_objects
    
    def combined_detection(self, frame):
        """Combine ML predictions with rule-based detection"""
        ml_prediction, ml_confidence = self.predict_behavior(frame)
        rule_score, detected_objects = self.rule_based_detection(frame)
        
        combined_score = rule_score
        detection_method = "Rule-Based"
        
        # Enhance with ML predictions if available and confident
        if ml_prediction and ml_confidence > self.confidence_threshold:
            detection_method = "ML + Rules"
            
            if ml_prediction in self.violence_classes:
                # Boost score for violence-related ML predictions
                ml_boost = int(ml_confidence * 50)
                combined_score += ml_boost
                
                # Update detected objects with ML info
                detected_objects.append((10, 10, 100, 30, f"ML: {ml_prediction}", (255, 0, 0)))
        
        # Store in history for temporal analysis
        self.prediction_history.append(combined_score)
        
        # Check for sustained aggressive behavior
        if len(self.prediction_history) >= 5:
            recent_scores = list(self.prediction_history)[-5:]
            avg_score = np.mean(recent_scores)
            if avg_score > 60:  # Sustained aggression
                combined_score = max(combined_score, 80)
        
        return min(combined_score, 100), detected_objects, detection_method, ml_prediction
    
    def add_alert(self, message, severity="MEDIUM"):
        """Add alert to log with cooldown"""
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            timestamp = time.strftime('%H:%M:%S')
            alert_msg = f"{timestamp} - {severity} - {message}"
            self.alerts_log.append(alert_msg)
            self.last_alert_time = current_time
            
            # Print alert with appropriate emoji
            if severity == "HIGH":
                print(f"🚨 {alert_msg}")
            elif severity == "MEDIUM":
                print(f"⚠️  {alert_msg}")
            else:
                print(f"📢 {alert_msg}")
            
            return True
        return False
    
    def draw_detection_interface(self, frame, assault_score, objects, method, ml_prediction):
        """Draw comprehensive detection interface"""
        h, w = frame.shape[:2]
        
        # Determine threat level and colors
        if assault_score >= 80:
            status_color = (0, 0, 255)  # Red
            status_text = "ASSAULT DETECTED! 🚨"
            alert_level = "HIGH"
            # Flashing warning
            if int(time.time() * 2) % 2 == 0:
                cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 25)
        elif assault_score >= 60:
            status_color = (0, 165, 255)  # Orange
            status_text = "SUSPICIOUS ACTIVITY ⚠️"
            alert_level = "MEDIUM"
        elif assault_score >= 30:
            status_color = (0, 255, 255)  # Yellow
            status_text = "ELEVATED ACTIVITY 📢"
            alert_level = "LOW"
        else:
            status_color = (0, 255, 0)  # Green
            status_text = "NORMAL ✅"
            alert_level = "NONE"
        
        # Main status bar
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status_text}", (20, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Threat Score: {assault_score}% | Method: {method}", (20, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # ML prediction info
        if ml_prediction:
            cv2.putText(frame, f"ML: {ml_prediction}", (20, 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Threat level meter
        meter_x, meter_y = w - 220, 20
        meter_width = 200
        meter_height = 25
        
        # Background
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + meter_width, meter_y + meter_height), 
                     (100, 100, 100), -1)
        
        # Fill based on threat level
        fill_width = int(meter_width * (assault_score / 100))
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + fill_width, meter_y + meter_height), 
                     status_color, -1)
        
        # Border and labels
        cv2.rectangle(frame, (meter_x, meter_y), 
                     (meter_x + meter_width, meter_y + meter_height), 
                     (255, 255, 255), 1)
        cv2.putText(frame, "THREAT LEVEL", (meter_x, meter_y - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw detected objects with labels
        for (x, y, w_obj, h_obj, label, color) in objects:
            cv2.rectangle(frame, (x, y), (x + w_obj, y + h_obj), color, 2)
            cv2.putText(frame, label, (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        # Recent alerts sidebar
        alert_bg_width = 400
        cv2.rectangle(frame, (w - alert_bg_width, 100), (w, h), (0, 0, 0), -1)
        cv2.rectangle(frame, (w - alert_bg_width, 100), (w, h), (50, 50, 50), 1)
        
        cv2.putText(frame, "RECENT ALERTS:", (w - alert_bg_width + 10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_offset = 140
        for alert in self.alerts_log[-6:]:  # Show last 6 alerts
            # BIKIN WARNA
            if "HIGH" in alert:
                color = (0, 0, 255)
            elif "MEDIUM" in alert:
                color = (0, 165, 255)
            else:
                color = (0, 255, 255)
            
            cv2.putText(frame, alert, (w - alert_bg_width + 10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
            y_offset += 20
        
        # Trigger alerts based on threat level
        if assault_score >= 80:
            self.add_alert(f"High threat detected! Score: {assault_score}%", "HIGH")
        elif assault_score >= 60:
            self.add_alert(f"Suspicious activity detected", "MEDIUM")
        
        # Instructions
        instructions = [
            "DEMONSTRATE:",
            "- Aggressive poses",
            "- Rapid movements", 
            "- Fighting stances",
            "- Press 'Q' to quit"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(frame, instruction, (10, h - 80 + i*20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        return frame, alert_level
    
    def run_detection(self):
        """Main detection loop"""
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("\n🎥 Starting real-time assault detection...")
        print("💡 Demonstration Guide:")
        print("   • Normal movement = Green (0-30%)")
        print("   • Elevated activity = Yellow (30-60%)") 
        print("   • Suspicious activity = Orange (60-80%)")
        print("   • Assault detected = Red (80-100%)")
        print("   • ML predictions show in blue text")
        print("Press 'Q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame = cv2.flip(frame, 1)
                
                # Run combined detection
                assault_score, objects, method, ml_prediction = self.combined_detection(frame)
                
                # Draw interface
                frame, alert_level = self.draw_detection_interface(
                    frame, assault_score, objects, method, ml_prediction
                )
                
                # Display frame
                cv2.imshow('Assault Detection System - Press Q to quit', frame)
                
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    
        except Exception as e:
            print(f"❌ Detection error: {e}")
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.show_session_summary()
    
    def show_session_summary(self):
        """Show detection session summary"""
        print("\n" + "=" * 60)
        print("📊 DETECTION SESSION SUMMARY")
        print("=" * 60)
        
        print(f"Total alerts triggered: {len(self.alerts_log)}")
        
        if self.alerts_log:
            print("\nRecent alerts:")
            for alert in self.alerts_log[-10:]:
                print(f"  • {alert}")
        else:
            print("\n✅ No alerts triggered during session")
        
        print(f"\n🔧 System Configuration:")
        print(f"  • Model: {'Loaded' if self.model else 'Rule-based only'}")
        print(f"  • Detection: Combined ML + Rule-based")
        print(f"  • Alert threshold: {self.assault_threshold}")
        print(f"  • Confidence threshold: {self.confidence_threshold}")

if __name__ == "__main__":
    
    model_path = None
    models_dir = "models"
    
    if os.path.exists(models_dir):
        model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]
        if model_files:
            # Get the most recent model
            model_files.sort(reverse=True)
            model_path = os.path.join(models_dir, model_files[0])
    
    # Initialize and run the detection system
    detector = AssaultDetectionSystem(model_path=model_path)
    detector.run_detection()

    # assault_detection_landscape.py
import cv2, numpy as np, tensorflow as tf, time, os
from collections import deque

class AssaultDetectionSystem:
    def __init__(self, model_path=None):
        self.model = keras.models.load_model(model_path) if model_path and os.path.exists(model_path) else None
        self.assault_threshold, self.confidence_threshold, self.alert_cooldown = 0.7, 0.6, 5
        self.last_alert_time, self.detection_history, self.alerts_log, self.prediction_history = 0, deque(maxlen=10), [], deque(maxlen=15)
        self.prev_frame, self.motion_history = None, deque(maxlen=10)
        self.classes = ["normal_standing", "normal_walking", "normal_sitting", "aggressive_pose", "fighting_stance", "raised_arms", "rapid_movement", "pushing_motion", "grabbing_motion", "falling_down"]
        self.violence_classes = ["aggressive_pose", "fighting_stance", "raised_arms", "pushing_motion", "grabbing_motion"]
        print("🚨 Assault Detection System Initialized | 📊 Combined ML + Rule-based Detection")

    def preprocess_frame(self, frame): processed = cv2.resize(frame, (224, 224)); processed = processed.astype(np.float32) / 255.0; return np.expand_dims(processed, axis=0)
    def predict_behavior(self, frame):
        if self.model is None: return None, 0
        try: processed_frame = self.preprocess_frame(frame); predictions = self.model.predict(processed_frame, verbose=0); predicted_class_idx = np.argmax(predictions[0]); confidence = np.max(predictions[0]); return self.classes[predicted_class_idx], confidence
        except Exception as e: print(f"❌ Prediction error: {e}"); return None, 0

    def rule_based_detection(self, frame):
        aggression_score, detected_objects = 0, []; gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY); gray = cv2.GaussianBlur(gray, (21, 21), 0)
        if self.prev_frame is None: self.prev_frame = gray; return aggression_score, detected_objects
        frame_diff = cv2.absdiff(self.prev_frame, gray); _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY); thresh = cv2.dilate(thresh, None, iterations=2)
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 5000:
                x, y, w, h = cv2.boundingRect(contour); aspect_ratio = w / h if h > 0 else 0
                if aspect_ratio > 1.8: aggression_score += 40; detected_objects.append((x, y, w, h, "AGGRESSIVE_STANCE", (0, 165, 255)))
                elif aspect_ratio < 0.4: aggression_score += 35; detected_objects.append((x, y, w, h, "RAISED_ARMS", (0, 0, 255)))
                else: detected_objects.append((x, y, w, h, "PERSON", (0, 255, 0)))
                self.motion_history.append((x, y, w, h))
                if len(self.motion_history) >= 3:
                    recent_motions = list(self.motion_history)[-3:]; x_movements = [pos[0] for pos in recent_motions]
                    if len(x_movements) >= 2:
                        movement_speed = abs(x_movements[-1] - x_movements[0])
                        if movement_speed > 50: aggression_score += 25; detected_objects.append((x, y, w, h, "RAPID_MOVEMENT", (0, 255, 255)))
        self.prev_frame = gray; return min(aggression_score, 100), detected_objects

    def combined_detection(self, frame):
        ml_prediction, ml_confidence = self.predict_behavior(frame); rule_score, detected_objects = self.rule_based_detection(frame)
        combined_score = rule_score; detection_method = "Rule-Based"
        if ml_prediction and ml_confidence > self.confidence_threshold:
            detection_method = "ML + Rules"
            if ml_prediction in self.violence_classes: ml_boost = int(ml_confidence * 50); combined_score += ml_boost; detected_objects.append((10, 10, 100, 30, f"ML: {ml_prediction}", (255, 0, 0)))
        self.prediction_history.append(combined_score)
        if len(self.prediction_history) >= 5: recent_scores = list(self.prediction_history)[-5:]; avg_score = np.mean(recent_scores); combined_score = max(combined_score, 80) if avg_score > 60 else combined_score
        return min(combined_score, 100), detected_objects, detection_method, ml_prediction

    def add_alert(self, message, severity="MEDIUM"):
        current_time = time.time()
        if current_time - self.last_alert_time > self.alert_cooldown:
            timestamp = time.strftime('%H:%M:%S'); alert_msg = f"{timestamp} - {severity} - {message}"; self.alerts_log.append(alert_msg); self.last_alert_time = current_time
            if severity == "HIGH": print(f"🚨 {alert_msg}")
            elif severity == "MEDIUM": print(f"⚠️  {alert_msg}")
            else: print(f"📢 {alert_msg}")
            return True
        return False

    def draw_detection_interface(self, frame, assault_score, objects, method, ml_prediction):
        h, w = frame.shape[:2]
        if assault_score >= 80: status_color, status_text, alert_level = (0, 0, 255), "ASSAULT DETECTED! 🚨", "HIGH"
        elif assault_score >= 60: status_color, status_text, alert_level = (0, 165, 255), "SUSPICIOUS ACTIVITY ⚠️", "MEDIUM"
        elif assault_score >= 30: status_color, status_text, alert_level = (0, 255, 255), "ELEVATED ACTIVITY 📢", "LOW"
        else: status_color, status_text, alert_level = (0, 255, 0), "NORMAL ✅", "NONE"
        if assault_score >= 80 and int(time.time() * 2) % 2 == 0: cv2.rectangle(frame, (0, 0), (w, h), (0, 0, 255), 25)
        cv2.rectangle(frame, (0, 0), (w, 100), (0, 0, 0), -1)
        cv2.putText(frame, f"STATUS: {status_text}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
        cv2.putText(frame, f"Threat Score: {assault_score}% | Method: {method}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        if ml_prediction: cv2.putText(frame, f"ML: {ml_prediction}", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        meter_x, meter_y, meter_width, meter_height = w - 220, 20, 200, 25
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (100, 100, 100), -1)
        fill_width = int(meter_width * (assault_score / 100)); cv2.rectangle(frame, (meter_x, meter_y), (meter_x + fill_width, meter_y + meter_height), status_color, -1)
        cv2.rectangle(frame, (meter_x, meter_y), (meter_x + meter_width, meter_y + meter_height), (255, 255, 255), 1)
        cv2.putText(frame, "THREAT LEVEL", (meter_x, meter_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        for (x, y, w_obj, h_obj, label, color) in objects: cv2.rectangle(frame, (x, y), (x + w_obj, y + h_obj), color, 2); cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        alert_bg_width = 400; cv2.rectangle(frame, (w - alert_bg_width, 100), (w, h), (0, 0, 0), -1); cv2.rectangle(frame, (w - alert_bg_width, 100), (w, h), (50, 50, 50), 1)
        cv2.putText(frame, "RECENT ALERTS:", (w - alert_bg_width + 10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset = 140
        for alert in self.alerts_log[-6:]:
            color = (0, 0, 255) if "HIGH" in alert else (0, 165, 255) if "MEDIUM" in alert else (0, 255, 255)
            cv2.putText(frame, alert, (w - alert_bg_width + 10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1); y_offset += 20
        if assault_score >= 80: self.add_alert(f"High threat detected! Score: {assault_score}%", "HIGH")
        elif assault_score >= 60: self.add_alert(f"Suspicious activity detected", "MEDIUM")
        instructions = ["DEMONSTRATE:", "- Aggressive poses", "- Rapid movements", "- Fighting stances", "- Press 'Q' to quit"]
        for i, instruction in enumerate(instructions): cv2.putText(frame, instruction, (10, h - 80 + i*20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        return frame, alert_level

    def run_detection(self):
        cap = cv2.VideoCapture(0); cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        print("\n🎥 Starting Real-time Detection | 💡 Guide: Green(0-30%) = Normal, Yellow(30-60%) = Elevated, Orange(60-80%) = Suspicious, Red(80-100%) = Assault | Press 'Q' to quit\n")
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)
                assault_score, objects, method, ml_prediction = self.combined_detection(frame)
                frame, alert_level = self.draw_detection_interface(frame, assault_score, objects, method, ml_prediction)
                cv2.imshow('Assault Detection System - Press Q to quit', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        except Exception as e:
            print(f"❌ Detection error: {e}")
        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.show_session_summary()

    def show_session_summary(self):
        print("\n" + "=" * 60); print("📊 DETECTION SESSION SUMMARY"); print("=" * 60); print(f"Total alerts triggered: {len(self.alerts_log)}")
        if self.alerts_log: print("\nRecent alerts:"); [print(f"  • {alert}") for alert in self.alerts_log[-10:]]
        else: print("\n✅ No alerts triggered during session")
        print(f"\n🔧 System Configuration: Model: {'Loaded' if self.model else 'Rule-based only'} | Detection: Combined ML + Rule-based | Alert threshold: {self.assault_threshold} | Confidence threshold: {self.confidence_threshold}")

if __name__ == "__main__":
    model_path = None; models_dir = "models"
    if os.path.exists(models_dir): model_files = [f for f in os.listdir(models_dir) if f.endswith('.h5')]; model_path = os.path.join(models_dir, model_files[0]) if model_files else None
    detector = AssaultDetectionSystem(model_path=model_path); detector.run_detection()