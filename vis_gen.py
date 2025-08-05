import cv2
import numpy as np
from collections import deque
import time

class ObjectMovementTracker:
    def __init__(self):
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Movement tracking parameters
        self.prev_frame = None
        self.movement_history = deque(maxlen=30)  # Store last 30 movement scores
        self.object_history = {}  # Track individual objects
        self.next_object_id = 0
        
        # Motion detection parameters
        self.min_contour_area = 100  # Further reduced minimum area
        self.max_contour_area = 50000
        self.movement_threshold = 10  # Further reduced threshold for more sensitivity
        
        # Display settings
        self.debug_mode = False
        self.show_trajectories = True
        self.fullscreen_mode = False
        
        # Screen size options (more conservative resolutions)
        self.screen_sizes = {
            '1': (320, 240),    # Small - Fast performance
            '2': (640, 480),    # Medium - Balanced
            '3': (800, 600),    # Large - Better detail
            '4': (1024, 768),   # Large - Good detail
            '5': (1280, 720)    # HD - High detail (if supported)
        }
        self.current_size_index = '2'  # Default to medium size
        self.update_camera_resolution()
        
        print("Object Movement Tracker initialized")
        print(f"Current resolution: {self.screen_sizes[self.current_size_index][0]}x{self.screen_sizes[self.current_size_index][1]}")
        print("\nAvailable screen sizes:")
        for key, (width, height) in self.screen_sizes.items():
            print(f"  {key} - {width}x{height}")
        print("\nControls:")
        print("  'd' - Toggle debug mode")
        print("  't' - Toggle trajectory display")
        print("  '+'/'-' - Adjust motion sensitivity")
        print("  '1'-'5' - Change screen size")
        print("  'f' - Toggle fullscreen")
        print("  'q' - Quit")
        print("  'r' - Reset tracking")
    
    def calculate_movement_score(self, contour, prev_positions):
        """Calculate movement score (0-1) for a contour based on its movement history"""
        if not prev_positions:
            return 0.0
        
        # Get current center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return 0.0
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        current_pos = (cx, cy)
        
        # Calculate movement from previous positions
        total_movement = 0
        for prev_pos in prev_positions:
            distance = np.sqrt((current_pos[0] - prev_pos[0])**2 + (current_pos[1] - prev_pos[1])**2)
            total_movement += distance
        
        # Normalize movement score (0-1)
        # Higher movement = higher score
        max_possible_movement = len(prev_positions) * 100  # Arbitrary max distance
        movement_score = min(1.0, total_movement / max_possible_movement)
        
        return movement_score
    
    def detect_moving_objects(self, frame):
        """Detect moving objects in the frame"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)  # Reduced blur kernel
        
        if self.prev_frame is None:
            self.prev_frame = gray
            return []
        
        # Calculate frame difference
        frame_diff = cv2.absdiff(self.prev_frame, gray)
        thresh = cv2.threshold(frame_diff, self.movement_threshold, 255, cv2.THRESH_BINARY)[1]
        
        # Morphological operations to reduce noise (less aggressive)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))  # Smaller kernel
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        moving_objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if self.min_contour_area < area < self.max_contour_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(contour)
                
                # Calculate center
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                    
                    moving_objects.append({
                        'contour': contour,
                        'center': (cx, cy),
                        'bbox': (x, y, w, h),
                        'area': area
                    })
        
        self.prev_frame = gray
        return moving_objects
    
    def track_objects(self, moving_objects):
        """Track individual objects and assign IDs"""
        current_objects = {}
        
        for obj in moving_objects:
            obj_center = obj['center']
            best_match_id = None
            min_distance = float('inf')
            
            # Find closest existing object
            for obj_id, tracked_obj in self.object_history.items():
                if tracked_obj['positions']:
                    last_pos = tracked_obj['positions'][-1]
                    distance = np.sqrt((obj_center[0] - last_pos[0])**2 + (obj_center[1] - last_pos[1])**2)
                    
                    if distance < 50 and distance < min_distance:  # Max 50 pixels movement
                        min_distance = distance
                        best_match_id = obj_id
            
            if best_match_id is not None:
                # Update existing object
                self.object_history[best_match_id]['positions'].append(obj_center)
                self.object_history[best_match_id]['bbox'] = obj['bbox']
                self.object_history[best_match_id]['area'] = obj['area']
                current_objects[best_match_id] = self.object_history[best_match_id]
            else:
                # Create new object
                new_id = self.next_object_id
                self.next_object_id += 1
                
                self.object_history[new_id] = {
                    'positions': [obj_center],
                    'bbox': obj['bbox'],
                    'area': obj['area'],
                    'color': tuple(np.random.randint(0, 255, 3).tolist())
                }
                current_objects[new_id] = self.object_history[new_id]
        
        # Clean up old objects (not seen for 30 frames)
        for obj_id in list(self.object_history.keys()):
            if obj_id not in current_objects:
                if len(self.object_history[obj_id]['positions']) > 30:
                    del self.object_history[obj_id]
        
        return current_objects
    
    def calculate_object_movement_scores(self, tracked_objects):
        """Calculate movement scores for all tracked objects"""
        scores = {}
        
        for obj_id, obj_data in tracked_objects.items():
            positions = obj_data['positions']
            if len(positions) > 1:
                # Calculate movement score based on recent positions
                recent_positions = positions[-10:]  # Last 10 positions
                movement_score = self.calculate_movement_score_from_positions(recent_positions)
                scores[obj_id] = movement_score
            else:
                scores[obj_id] = 0.0
        
        return scores
    
    def calculate_movement_score_from_positions(self, positions):
        """Calculate movement score from a list of positions"""
        if len(positions) < 2:
            return 0.0
        
        total_distance = 0
        for i in range(1, len(positions)):
            prev_pos = positions[i-1]
            curr_pos = positions[i]
            distance = np.sqrt((curr_pos[0] - prev_pos[0])**2 + (curr_pos[1] - prev_pos[1])**2)
            total_distance += distance
        
        # Normalize to 0-1 scale
        avg_distance = total_distance / (len(positions) - 1)
        max_expected_distance = 50  # Maximum expected movement per frame
        movement_score = min(1.0, avg_distance / max_expected_distance)
        
        return movement_score
    
    def draw_objects_and_scores(self, frame, tracked_objects, movement_scores):
        """Draw tracked objects and their movement scores"""
        for obj_id, obj_data in tracked_objects.items():
            if obj_id in movement_scores:
                score = movement_scores[obj_id]
                bbox = obj_data['bbox']
                color = obj_data['color']
                
                # Draw bounding box
                x, y, w, h = bbox
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                
                # Draw object ID and movement score
                label = f"ID:{obj_id} Score:{score:.2f}"
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw trajectory if enabled
                if self.show_trajectories and len(obj_data['positions']) > 1:
                    positions = obj_data['positions']
                    for i in range(1, len(positions)):
                        cv2.line(frame, positions[i-1], positions[i], color, 2)
                
                # Draw movement indicator
                if score > 0.5:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 255, 0), -1)  # Green for high movement
                elif score > 0.2:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 255, 255), -1)  # Yellow for medium movement
                else:
                    cv2.circle(frame, (x + w//2, y + h//2), 5, (0, 0, 255), -1)  # Red for low movement
    
    def draw_debug_info(self, frame, moving_objects, tracked_objects, movement_scores):
        """Draw debug information"""
        # Draw motion mask
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (15, 15), 0)  # Match the detection method
        
        if self.prev_frame is not None:
            frame_diff = cv2.absdiff(self.prev_frame, gray)
            # Use a lower threshold for debug display to make motion more visible
            debug_threshold = max(5, self.movement_threshold - 5)  # More sensitive for debug
            thresh = cv2.threshold(frame_diff, debug_threshold, 255, cv2.THRESH_BINARY)[1]
            
            # Apply same morphological operations as detection
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
            thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
            
            # Resize for display - make debug windows larger for better visibility
            height, width = frame.shape[:2]
            debug_width = width // 2  # Half the main frame width
            debug_height = height // 2  # Half the main frame height
            
            motion_mask_resized = cv2.resize(thresh, (debug_width, debug_height))
            
            # Enhance visibility by making motion areas more prominent
            motion_mask_enhanced = cv2.cvtColor(motion_mask_resized, cv2.COLOR_GRAY2BGR)
            # Make white areas (motion) more visible with bright colors
            motion_mask_enhanced[motion_mask_resized == 255] = [0, 255, 255]  # Bright cyan for motion
            # Also make gray areas slightly visible for context
            motion_mask_enhanced[motion_mask_resized == 0] = [20, 20, 20]  # Dark gray for no motion
            
            # Debug: Add some test motion to verify display is working
            if np.sum(motion_mask_resized) == 0:  # If no motion detected, add a test pattern
                cv2.circle(motion_mask_enhanced, (debug_width//2, debug_height//2), 20, [0, 255, 255], -1)
                cv2.putText(motion_mask_enhanced, "NO MOTION", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, [0, 255, 255], 1)
            
            # Overlay motion mask
            frame[10:10+debug_height, width-debug_width-10:width-10] = motion_mask_enhanced
            
            # Add border
            cv2.rectangle(frame, (width-debug_width-10, 10), (width-10, 10+debug_height), (255, 255, 255), 2)
            cv2.putText(frame, f"Motion Mask (Threshold: {self.movement_threshold})", (width-debug_width-10, 25), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Show raw motion detection (before filtering) in bottom-left
            raw_diff = cv2.absdiff(self.prev_frame, gray)
            raw_thresh = cv2.threshold(raw_diff, debug_threshold, 255, cv2.THRESH_BINARY)[1]
            
            raw_width = width // 3
            raw_height = height // 3
            raw_resized = cv2.resize(raw_thresh, (raw_width, raw_height))
            
            # Enhance raw motion visibility
            raw_enhanced = cv2.cvtColor(raw_resized, cv2.COLOR_GRAY2BGR)
            # Make white areas (motion) more visible with bright green
            raw_enhanced[raw_resized == 255] = [0, 255, 0]  # Bright green for raw motion
            # Also make gray areas slightly visible for context
            raw_enhanced[raw_resized == 0] = [20, 20, 20]  # Dark gray for no motion
            
            # Debug: Add some test motion to verify display is working
            if np.sum(raw_resized) == 0:  # If no motion detected, add a test pattern
                cv2.circle(raw_enhanced, (raw_width//2, raw_height//2), 15, [0, 255, 0], -1)
                cv2.putText(raw_enhanced, "NO MOTION", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, [0, 255, 0], 1)
            
            # Overlay raw motion mask
            frame[height-raw_height-10:height-10, 10:10+raw_width] = raw_enhanced
            
            # Add border
            cv2.rectangle(frame, (10, height-raw_height-10), (10+raw_width, height-10), (0, 255, 0), 2)
            cv2.putText(frame, "Raw Motion", (10, height-raw_height+15), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw statistics
        current_width, current_height = self.screen_sizes[self.current_size_index]
        stats_text = [
            f"Objects tracked: {len(tracked_objects)}",
            f"Moving objects: {len(moving_objects)}",
            f"Avg movement: {np.mean(list(movement_scores.values())):.2f}" if movement_scores else "0.00",
            f"Motion threshold: {self.movement_threshold}",
            f"Min contour area: {self.min_contour_area}",
            f"Resolution: {current_width}x{current_height}",
            f"Press +/- to adjust sensitivity"
        ]
        
        y_offset = 30
        for text in stats_text:
            cv2.putText(frame, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 25
    
    def update_camera_resolution(self):
        """Update camera resolution based on current size index"""
        width, height = self.screen_sizes[self.current_size_index]
        
        # Release and recreate camera to ensure clean resolution change
        self.cap.release()
        self.cap = cv2.VideoCapture(0)
        
        # Set camera properties
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        # Give camera time to adjust
        time.sleep(0.2)
        
        # Verify the resolution was set correctly
        actual_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        # If the camera doesn't support the requested resolution, fall back to a supported one
        if abs(actual_width - width) > 50 or abs(actual_height - height) > 50:
            print(f"Warning: Camera doesn't support {width}x{height}, using {int(actual_width)}x{int(actual_height)}")
            # Update the screen_sizes dictionary with the actual supported resolution
            self.screen_sizes[self.current_size_index] = (int(actual_width), int(actual_height))
            # Use the actual width for parameter adjustment
            width = int(actual_width)
        
        # Adjust motion detection parameters based on resolution
        if width <= 320:
            self.min_contour_area = 30
            self.movement_threshold = 8
        elif width <= 640:
            self.min_contour_area = 100
            self.movement_threshold = 10
        elif width <= 800:
            self.min_contour_area = 150
            self.movement_threshold = 12
        else:
            self.min_contour_area = 200
            self.movement_threshold = 15
    
    def change_screen_size(self, size_index):
        """Change screen size"""
        if size_index in self.screen_sizes:
            try:
                self.current_size_index = size_index
                self.update_camera_resolution()
                # Reset prev_frame to avoid size mismatch
                self.prev_frame = None
                width, height = self.screen_sizes[size_index]
                print(f"Screen size changed to: {width}x{height}")
                print(f"Motion threshold: {self.movement_threshold}, Min contour area: {self.min_contour_area}")
            except Exception as e:
                print(f"Failed to change to size {size_index}: {e}")
                # Fall back to a safe resolution
                self.current_size_index = '2'
                self.update_camera_resolution()
                self.prev_frame = None
        else:
            print(f"Invalid size option. Available: 1-5")
    
    def toggle_fullscreen(self):
        """Toggle fullscreen mode"""
        self.fullscreen_mode = not self.fullscreen_mode
        if self.fullscreen_mode:
            cv2.setWindowProperty('Object Movement Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            print("Fullscreen mode: ON")
        else:
            cv2.setWindowProperty('Object Movement Tracker', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
            print("Fullscreen mode: OFF")
    
    def adjust_motion_sensitivity(self, direction):
        """Adjust motion detection sensitivity"""
        if direction == 'increase':
            self.movement_threshold = max(1, self.movement_threshold - 2)  # Allow threshold to go down to 1
            print(f"Motion threshold decreased to: {self.movement_threshold}")
        elif direction == 'decrease':
            self.movement_threshold = min(50, self.movement_threshold + 2)
            print(f"Motion threshold increased to: {self.movement_threshold}")
    
    def reset_tracking(self):
        """Reset all tracking data"""
        self.object_history = {}
        self.next_object_id = 0
        self.prev_frame = None
        print("Tracking reset")
    
    def run(self):
        """Main loop"""
        print("Starting Object Movement Tracker...")
        
        # Create window with proper properties
        cv2.namedWindow('Object Movement Tracker', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Object Movement Tracker', 640, 480)
        
        while True:
            try:
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame - trying to reconnect...")
                    # Try to reconnect to camera
                    self.cap.release()
                    self.cap = cv2.VideoCapture(0)
                    self.update_camera_resolution()
                    continue
            except Exception as e:
                print(f"Camera read error: {e} - trying to reconnect...")
                self.cap.release()
                self.cap = cv2.VideoCapture(0)
                self.update_camera_resolution()
                continue
            
            # Detect moving objects
            moving_objects = self.detect_moving_objects(frame)
            
            # Track objects
            tracked_objects = self.track_objects(moving_objects)
            
            # Calculate movement scores
            movement_scores = self.calculate_object_movement_scores(tracked_objects)
            
            # Draw everything
            if self.debug_mode:
                self.draw_debug_info(frame, moving_objects, tracked_objects, movement_scores)
            
            self.draw_objects_and_scores(frame, tracked_objects, movement_scores)
            
            # Add instructions
            cv2.putText(frame, "Press 'd' for debug, 't' for trajectories, '+/-' for sensitivity, '1-5' for size, 'f' for fullscreen, 'r' to reset, 'q' to quit", 
                       (10, frame.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Display frame
            cv2.imshow('Object Movement Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('d'):
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('t'):
                self.show_trajectories = not self.show_trajectories
                print(f"Trajectory display: {'ON' if self.show_trajectories else 'OFF'}")
            elif key == ord('f'):
                self.toggle_fullscreen()
            elif key == ord('+') or key == ord('='):
                self.adjust_motion_sensitivity('increase')
            elif key == ord('-') or key == ord('_'):
                self.adjust_motion_sensitivity('decrease')
            elif key in [ord('1'), ord('2'), ord('3'), ord('4'), ord('5')]:
                self.change_screen_size(chr(key))
            elif key == ord('r'):
                self.reset_tracking()
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = ObjectMovementTracker()
    tracker.run() 