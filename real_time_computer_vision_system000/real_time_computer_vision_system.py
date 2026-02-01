import cv2  # OpenCV for computer vision tasks
import numpy as np
import os # For file system operations
import time # For performance tracking
from threading import Lock

class VisionSystem:
    def __init__(self):
        # yaha sa camera start karna hai
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("Error: Could not open camera!")
            exit()
        
        # yaha sa camera properties set karna hai
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # yaha sa mode set karna hai
        self.mode = "color"
        self.running = True
        
        # yaha sa face detector load karna hai with error handling
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            print(f"Error: Cascade file not found at {cascade_path}")
            # Try alternative path
            cascade_path = 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                print("Downloading cascade file...")
                import urllib.request
                url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"
                urllib.request.urlretrieve(url, cascade_path)
        
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        if self.face_cascade.empty():
            print("Error: Failed to load face cascade classifier!")
            exit()
        
        # Thread lock for thread safety in web mode
        self.capture_lock = Lock()
        
        # yaha sa color ranges define karna hai
        self.color_ranges = {
            "Red": ([0, 100, 100], [10, 255, 255]),
            "Blue": ([100, 100, 100], [130, 255, 255]),
            "Green": ([40, 100, 100], [80, 255, 255]),
            "Yellow": ([20, 100, 100], [30, 255, 255]),
            "Purple": ([130, 100, 100], [160, 255, 255]),
            "Orange": ([10, 100, 100], [20, 255, 255]),
            "Cyan": ([85, 100, 100], [100, 255, 255]),
            "Pink": ([160, 100, 100], [180, 255, 255]),
        }
        
        self.color_dict = {
            "Red": (0, 0, 255),
            "Blue": (255, 0, 0),
            "Green": (0, 255, 0),
            "Yellow": (0, 255, 255),
            "Purple": (128, 0, 128),
            "Orange": (0, 165, 255),
            "Cyan": (255, 255, 0),
            "Pink": (147, 20, 255),
        }
        
        # yaha sa motion detection variables set karna hai
        self.prev_gray = None
        self.motion_threshold = 25
        self.min_motion_area = 500
        
        # yaha sa face capture hoga
        self.face_count = 0
        self.capture_folder = "captured_faces"
        if not os.path.exists(self.capture_folder):
            os.makedirs(self.capture_folder)
        
        # Performance tracking
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()
        
        # Display settings
        self.show_help = True
        self.show_fps = True
        
        # Web mode flag
        self.web_mode = False
        
        print("\n" + "="*60)
        print("VISION PROCESSING SYSTEM - REAL-TIME MODE SWITCHING")
        print("="*60)
    
    def calculate_fps(self):
        """Calculate frames per second"""
        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 1.0:  # Update FPS every second
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()
    
    def display_info(self, frame):
        """Display mode and control information on frame"""
        # yaha sa mode colors define karna hai
        mode_colors = {
            "color": (0, 165, 255),    # Orange
            "face": (0, 255, 0),       # Green
            "capture": (255, 0, 255),  # Magenta
            "motion": (255, 255, 0),   # Cyan
        }
        
        color = mode_colors.get(self.mode, (255, 255, 255))
        
        # Mode title
        cv2.putText(frame, f"MODE: {self.mode.upper()}", (10, 30),
                   cv2.FONT_HERSHEY_DUPLEX, 0.8, color, 2)
        
        # FPS counter
        if self.show_fps:
            cv2.putText(frame, f"FPS: {self.fps:.1f}", (frame.shape[1] - 100, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # yaha sa help text ayega (toggle with 'h' key)
        if self.show_help:
            help_y = 60
            cv2.putText(frame, "CONTROLS:", (10, help_y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(frame, "1-Color 2-Face 3-Capture 4-Motion", (10, help_y + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(frame, "SPACE-Capture  h-Help  q-Quit", (10, help_y + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 255), 1)
            cv2.putText(frame, f"Faces Captured: {self.face_count}", (10, help_y + 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1)
        
        return frame
    
    def process_color(self, frame):
        """Process frame for color detection"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        detected_colors = []
        
        for color_name, (lower, upper) in self.color_ranges.items():
            lower_np = np.array(lower)
            upper_np = np.array(upper)
            
            mask = cv2.inRange(hsv, lower_np, upper_np)
            
            #yaha sa morphology operations karna hai to reduce noise in mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > 300:  # Minimum area threshold
                    x, y, w, h = cv2.boundingRect(contour)
                    color = self.color_dict.get(color_name, (255, 255, 255))
                    
                    # Draw rectangle
                    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                    
                    # Draw label with background for better visibility
                    label = f"{color_name} ({area:.0f})"
                    (text_width, text_height), _ = cv2.getTextSize(
                        label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
                    )
                    
                    # Draw background rectangle for text
                    cv2.rectangle(frame, (x, y - text_height - 5),
                                 (x + text_width, y), color, -1)
                    
                    # Draw text
                    cv2.putText(frame, label, (x, y - 5),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                    
                    detected_colors.append(color_name)
        
        # Display detected colors count
        if detected_colors:
            unique_colors = set(detected_colors)
            cv2.putText(frame, f"Colors: {len(unique_colors)}", (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def process_face(self, frame):
        """Process frame for face detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Equalize histogram for better face detection
        gray = cv2.equalizeHist(gray)
        
        # Detect faces with error handling
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(50, 50),  # Increased minimum size
                maxSize=(300, 300),  # Added maximum size
                flags=cv2.CASCADE_SCALE_IMAGE
            )
        except Exception as e:
            print(f"Face detection error: {e}")
            faces = []
        
        # Draw rectangles around faces
        for i, (x, y, w, h) in enumerate(faces):
            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw face label
            cv2.putText(frame, f"Face {i+1}", (x, y - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Optional: Draw face center point
            center_x = x + w // 2
            center_y = y + h // 2
            cv2.circle(frame, (center_x, center_y), 3, (0, 255, 255), -1)
        
        return frame
    
    def process_capture(self, frame):
        """Process frame for face capture mode"""
        # First, detect faces (same as face detection)
        frame = self.process_face(frame)
        
        # Add capture-specific instructions
        if self.show_help:
            if self.web_mode:
                cv2.putText(frame, "CLICK CAPTURE BUTTON ON WEB INTERFACE", 
                           (frame.shape[1] // 2 - 180, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "PRESS SPACE TO CAPTURE FACE", 
                           (frame.shape[1] // 2 - 120, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        return frame
    
    def capture_current_face(self, frame):
        """Capture the current face in frame"""
        with self.capture_lock:
            try:
                # Validate frame
                if frame is None or frame.size == 0:
                    print("⚠ Capture failed: Invalid frame")
                    return False
                
                # Convert to grayscale
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Equalize histogram
                gray = cv2.equalizeHist(gray)
                
                # Detect faces with safe parameters
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=6,
                    minSize=(60, 60),
                    maxSize=(300, 300)
                )
                
                print(f"Detected {len(faces)} face(s) for capture")
                
                if len(faces) == 1:
                    x, y, w, h = faces[0]
                    
                    # Validate face coordinates
                    if x < 0 or y < 0 or x + w > frame.shape[1] or y + h > frame.shape[0]:
                        print("⚠ Capture failed: Face outside frame bounds")
                        return False
                    
                    # Expand the capture area slightly
                    expand = 30
                    x = max(0, x - expand)
                    y = max(0, y - expand)
                    w = min(frame.shape[1] - x, w + 2 * expand)
                    h = min(frame.shape[0] - y, h + 2 * expand)
                    
                    # Extract face region
                    face_img = frame[y:y + h, x:x + w]
                    
                    if face_img.size > 0:
                        self.face_count += 1
                        timestamp = time.strftime("%Y%m%d_%H%M%S")
                        filename = f"{self.capture_folder}/face_{self.face_count}_{timestamp}.jpg"
                        
                        # Save the image with compression
                        cv2.imwrite(filename, face_img, [cv2.IMWRITE_JPEG_QUALITY, 95])
                        
                        print(f"✓ Face captured: {filename}")
                        return True
                    else:
                        print("⚠ Capture failed: Empty face region")
                        return False
                else:
                    print(f"⚠ Capture failed: Need exactly one clear face (found {len(faces)})")
                    return False
                    
            except Exception as e:
                print(f"⚠ Capture error: {str(e)}")
                return False
    
    def process_motion(self, frame):
        """Process frame for motion detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        if self.prev_gray is not None:
            # Calculate absolute difference
            diff = cv2.absdiff(self.prev_gray, gray)
            
            # Apply threshold
            _, thresh = cv2.threshold(diff, self.motion_threshold, 255, cv2.THRESH_BINARY)
            
            # Apply dilation to fill gaps
            kernel = np.ones((5, 5), np.uint8)
            thresh = cv2.dilate(thresh, kernel, iterations=2)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            motion_detected = False
            motion_area = 0
            
            for contour in contours:
                area = cv2.contourArea(contour)
                if area > self.min_motion_area:
                    motion_detected = True
                    motion_area += area
                    
                    # Draw bounding box
                    x, y, w, h = cv2.boundingRect(contour)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 2)
                    
                    # Draw motion label
                    cv2.putText(frame, "MOTION", (x, y - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Display motion status
            if motion_detected:
                status_color = (0, 0, 255)  # Red
                status_text = f"MOTION DETECTED! Area: {motion_area:.0f}"
            else:
                status_color = (0, 255, 0)  # Green
                status_text = "No motion"
            
            cv2.putText(frame, status_text, (10, frame.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2)
        
        # Update previous frame
        self.prev_gray = gray.copy()
        
        return frame
    
    def process_frame(self, frame):
        """Process frame based on current mode"""
        self.calculate_fps()
        
        # Process based on mode
        if self.mode == "color":
            frame = self.process_color(frame)
        elif self.mode == "face":
            frame = self.process_face(frame)
        elif self.mode == "capture":
            frame = self.process_capture(frame)
        elif self.mode == "motion":
            frame = self.process_motion(frame)
        
        # Add information overlay
        frame = self.display_info(frame)
        
        return frame
    
    def get_processed_frame(self):
        """Get a processed frame as JPEG bytes for web streaming"""
        ret, frame = self.cap.read()
        if not ret:
            return None
        
        # Process the frame
        processed_frame = self.process_frame(frame)
        
        # Encode as JPEG
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        if not ret:
            return None
        
        return buffer.tobytes()
    
    def run(self):
        """Main loop for standalone mode"""
        print("\nInitializing vision system...")
        print("\nKEY CONTROLS:")
        print("  '1' - Color Detection Mode")
        print("  '2' - Face Detection Mode")
        print("  '3' - Face Capture Mode (save images)")
        print("  '4' - Motion Detection Mode")
        print("  SPACE - Capture face (in Capture Mode only)")
        print("  'h' - Toggle help display")
        print("  'f' - Toggle FPS display")
        print("  'q' - Quit application")
        print("\nStarting in Color Detection mode...")
        
        # Initialize with first frame for motion detection
        ret, first_frame = self.cap.read()
        if ret:
            self.prev_gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        
        print("\nSystem ready! Press 'h' for controls.")
        
        while self.running:
            # Read frame from camera
            ret, frame = self.cap.read()
            if not ret:
                print("Error: Failed to capture frame!")
                break
            
            # Process the frame
            processed_frame = self.process_frame(frame)
            
            # Display the frame
            cv2.imshow('Vision Processing System', processed_frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):  # Quit
                break
            elif key == ord('1'):  # Color mode
                self.mode = "color"
                print(f"Mode changed: COLOR DETECTION")
            elif key == ord('2'):  # Face mode
                self.mode = "face"
                print(f"Mode changed: FACE DETECTION")
            elif key == ord('3'):  # Capture mode
                self.mode = "capture"
                self.web_mode = False
                print(f"Mode changed: FACE CAPTURE (Press SPACE to capture)")
            elif key == ord('4'):  # Motion mode
                self.mode = "motion"
                print(f"Mode changed: MOTION DETECTION")
            elif key == ord('h'):  # Toggle help
                self.show_help = not self.show_help
                print(f"Help display: {'ON' if self.show_help else 'OFF'}")
            elif key == ord('f'):  # Toggle FPS
                self.show_fps = not self.show_fps
                print(f"FPS display: {'ON' if self.show_fps else 'OFF'}")
            elif key == ord(' '):  # Space bar - capture face
                if self.mode == "capture":
                    self.capture_current_face(frame.copy())
            elif key == ord('+'):  # Increase motion sensitivity
                self.motion_threshold = min(50, self.motion_threshold + 5)
                print(f"Motion threshold: {self.motion_threshold}")
            elif key == ord('-'):  # Decrease motion sensitivity
                self.motion_threshold = max(5, self.motion_threshold - 5)
                print(f"Motion threshold: {self.motion_threshold}")
        
        # Cleanup
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        print("\n" + "="*60)
        print("SHUTTING DOWN VISION SYSTEM")
        print("="*60)
        print(f"Total faces captured: {self.face_count}")
        print(f"Images saved in: {os.path.abspath(self.capture_folder)}")
        print(f"Average FPS: {self.fps:.1f}")
        
        if self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()
        
        print("\nSystem shutdown complete. Goodbye!")
        print("="*60)

def main():
    """Main entry point"""
    try:
        vision_system = VisionSystem()
        vision_system.run()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()