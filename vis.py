import cv2
import numpy as np
import time
from collections import deque

class HumanPoseTracker:
    def __init__(self, max_history=15):
        """
        Initialize the human pose tracker with OpenCV-based detection
        
        Args:
            max_history (int): Maximum number of frames to keep in history for movement calculation
        """
        # Load pre-trained models
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_fullbody.xml')
        self.upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
        
        # Note: haarcascade_hand.xml is not available in OpenCV, so we'll use improved skin color and motion detection
        
        # Store pose history for movement calculation
        self.pose_history = deque(maxlen=max_history)
        self.frame_width = 320  # Reduced for speed
        self.frame_height = 240  # Reduced for speed
        
        # Define key body parts we want to track (enhanced)
        self.key_landmarks = {
            'face_center': None,
            'face_left_eye': None,
            'face_right_eye': None,
            'upper_body_center': None,
            'upper_body_left': None,
            'upper_body_right': None,
            'full_body_center': None,
            'full_body_left': None,
            'full_body_right': None,
            'left_arm_shoulder': None,
            'left_arm_elbow': None,
            'left_arm_wrist': None,
            'right_arm_shoulder': None,
            'right_arm_elbow': None,
            'right_arm_wrist': None,
            'left_leg_hip': None,
            'left_leg_knee': None,
            'left_leg_ankle': None,
            'right_leg_hip': None,
            'right_leg_knee': None,
            'right_leg_ankle': None,
            'left_hand_center': None,
            'left_hand_thumb': None,
            'left_hand_index': None,
            'right_hand_center': None,
            'right_hand_thumb': None,
            'right_hand_index': None
        }
        
        # Motion detection parameters
        self.prev_frame = None
        self.motion_threshold = 25
        
        # Performance optimization flags
        self.frame_skip = 2  # Process every 2nd frame
        self.frame_count = 0
        self.last_pose_data = {}
        
        # Debug mode for visualization
        self.debug_mode = False  # Set to True to see skin detection masks
        
        # Improved skin color detection with more restrictive ranges to avoid false positives
        # Primary range - very restrictive to avoid poster detection
        self.lower_skin = np.array([0, 50, 140], dtype=np.uint8)
        self.upper_skin = np.array([20, 160, 190], dtype=np.uint8)
        
        # Alternative ranges for different lighting conditions - very restrictive
        self.lower_skin2 = np.array([0, 45, 160], dtype=np.uint8)
        self.upper_skin2 = np.array([25, 140, 170], dtype=np.uint8)
        
        # Third range for very light skin - very restrictive
        self.lower_skin3 = np.array([0, 55, 180], dtype=np.uint8)
        self.upper_skin3 = np.array([30, 130, 180], dtype=np.uint8)
        
        # Fourth range for darker skin tones - very restrictive
        self.lower_skin4 = np.array([0, 60, 120], dtype=np.uint8)
        self.upper_skin4 = np.array([20, 120, 160], dtype=np.uint8)
    
    def normalize_coordinates(self, x, y):
        """
        Normalize coordinates to 0-1 range based on frame dimensions
        
        Args:
            x, y: Raw pixel coordinates
            
        Returns:
            tuple: Normalized coordinates (0-1)
        """
        norm_x = x / self.frame_width
        norm_y = y / self.frame_height
        return norm_x, norm_y
    
    def calculate_movement_score(self, current_pos, previous_pos):
        """
        Calculate movement score between two positions (0-1)
        
        Args:
            current_pos: Current normalized position (x, y)
            previous_pos: Previous normalized position (x, y)
            
        Returns:
            float: Movement score (0-1)
        """
        if previous_pos is None:
            return 0.0
        
        # Calculate Euclidean distance
        distance = np.sqrt((current_pos[0] - previous_pos[0])**2 + 
                          (current_pos[1] - previous_pos[1])**2)
        
        # Normalize to 0-1 range (assuming max movement is 1.0)
        movement_score = min(distance, 1.0)
        return movement_score
    
    def extract_pose_data(self, frame):
        """
        Extract pose landmarks and calculate normalized values using OpenCV
        
        Args:
            frame: Input video frame
            
        Returns:
            dict: Dictionary containing pose data with normalized values
        """
        # Skip frames for performance
        self.frame_count += 1
        if self.frame_count % self.frame_skip != 0:
            return self.last_pose_data
        
        pose_data = {}
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces with even stricter parameters to reduce false positives
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 6, minSize=(50, 50), maxSize=(180, 180))
        for (x, y, w, h) in faces:
            # Additional validation: check if face is in reasonable position (not too high or low)
            if y < self.frame_height * 0.15 or y > self.frame_height * 0.85:
                continue  # Skip faces too close to top or bottom
                
            # Face center
            center_x = x + w // 2
            center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(center_x, center_y)
            
            # Calculate movement score
            movement_score = 0.0
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if 'face_center' in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data['face_center']['x'], prev_data['face_center']['y'])
                    )
            
            pose_data['face_center'] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': center_x,
                'raw_y': center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h
            }
            
            # Face left and right points
            left_x, left_y = self.normalize_coordinates(x, center_y)
            right_x, right_y = self.normalize_coordinates(x + w, center_y)
            
            pose_data['face_left_eye'] = {
                'x': left_x,
                'y': left_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            
            pose_data['face_right_eye'] = {
                'x': right_x,
                'y': right_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x + w,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            break  # Only process first face for speed
        
        # Detect upper body with even stricter parameters
        upper_bodies = self.upper_body_cascade.detectMultiScale(gray, 1.05, 6, minSize=(80, 80), maxSize=(250, 250))
        for (x, y, w, h) in upper_bodies:
            # Additional validation: check if upper body is in reasonable position
            if y < self.frame_height * 0.1 or y > self.frame_height * 0.9:
                continue  # Skip detections too close to edges
                
            center_x = x + w // 2
            center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(center_x, center_y)
            
            movement_score = 0.0
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if 'upper_body_center' in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data['upper_body_center']['x'], prev_data['upper_body_center']['y'])
                    )
            
            pose_data['upper_body_center'] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': center_x,
                'raw_y': center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h
            }
            
            # Upper body left and right points
            left_x, left_y = self.normalize_coordinates(x, center_y)
            right_x, right_y = self.normalize_coordinates(x + w, center_y)
            
            pose_data['upper_body_left'] = {
                'x': left_x,
                'y': left_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            
            pose_data['upper_body_right'] = {
                'x': right_x,
                'y': right_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x + w,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            break  # Only process first upper body for speed
        
        # Detect full body with optimized parameters
        bodies = self.body_cascade.detectMultiScale(gray, 1.2, 3, minSize=(80, 80))
        for (x, y, w, h) in bodies:
            center_x = x + w // 2
            center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(center_x, center_y)
            
            movement_score = 0.0
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if 'full_body_center' in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data['full_body_center']['x'], prev_data['full_body_center']['y'])
                    )
            
            pose_data['full_body_center'] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': center_x,
                'raw_y': center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h
            }
            
            # Full body left and right points
            left_x, left_y = self.normalize_coordinates(x, center_y)
            right_x, right_y = self.normalize_coordinates(x + w, center_y)
            
            pose_data['full_body_left'] = {
                'x': left_x,
                'y': left_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            
            pose_data['full_body_right'] = {
                'x': right_x,
                'y': right_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': x + w,
                'raw_y': center_y,
                'raw_z': 0.0
            }
            break  # Only process first body for speed
        
        # Detect arms and legs based on body position
        if 'upper_body_center' in pose_data:
            self.detect_arms_and_legs(pose_data, gray)
        
        # Try multiple hand detection methods for better results
        self.detect_hands_adaptive(pose_data, frame)
        
        self.last_pose_data = pose_data
        return pose_data
    
    def detect_arms_and_legs(self, pose_data, gray):
        """
        Detect arm and leg positions based on body detection
        
        Args:
            pose_data: Dictionary containing pose data
            gray: Grayscale image
        """
        if 'upper_body_center' in pose_data:
            body_center = pose_data['upper_body_center']
            body_x = int(body_center['raw_x'])
            body_y = int(body_center['raw_y'])
            body_w = body_center.get('width', 100)
            body_h = body_center.get('height', 150)
            
            # Estimate arm positions based on body proportions
            # Left arm
            left_shoulder_x = body_x - body_w // 3
            left_shoulder_y = body_y - body_h // 4
            left_elbow_x = left_shoulder_x - body_w // 6
            left_elbow_y = left_shoulder_y + body_h // 6
            left_wrist_x = left_elbow_x - body_w // 8
            left_wrist_y = left_elbow_y + body_h // 6
            
            # Right arm
            right_shoulder_x = body_x + body_w // 3
            right_shoulder_y = body_y - body_h // 4
            right_elbow_x = right_shoulder_x + body_w // 6
            right_elbow_y = right_shoulder_y + body_h // 6
            right_wrist_x = right_elbow_x + body_w // 8
            right_wrist_y = right_elbow_y + body_h // 6
            
            # Estimate leg positions
            # Left leg
            left_hip_x = body_x - body_w // 4
            left_hip_y = body_y + body_h // 3
            left_knee_x = left_hip_x - body_w // 8
            left_knee_y = left_hip_y + body_h // 2
            left_ankle_x = left_knee_x - body_w // 12
            left_ankle_y = left_knee_y + body_h // 2
            
            # Right leg
            right_hip_x = body_x + body_w // 4
            right_hip_y = body_y + body_h // 3
            right_knee_x = right_hip_x + body_w // 8
            right_knee_y = right_hip_y + body_h // 2
            right_ankle_x = right_knee_x + body_w // 12
            right_ankle_y = right_knee_y + body_h // 2
            
            # Add arm data
            arm_points = [
                ('left_arm_shoulder', left_shoulder_x, left_shoulder_y),
                ('left_arm_elbow', left_elbow_x, left_elbow_y),
                ('left_arm_wrist', left_wrist_x, left_wrist_y),
                ('right_arm_shoulder', right_shoulder_x, right_shoulder_y),
                ('right_arm_elbow', right_elbow_x, right_elbow_y),
                ('right_arm_wrist', right_wrist_x, right_wrist_y)
            ]
            
            # Add leg data
            leg_points = [
                ('left_leg_hip', left_hip_x, left_hip_y),
                ('left_leg_knee', left_knee_x, left_knee_y),
                ('left_leg_ankle', left_ankle_x, left_ankle_y),
                ('right_leg_hip', right_hip_x, right_hip_y),
                ('right_leg_knee', right_knee_x, right_knee_y),
                ('right_leg_ankle', right_ankle_x, right_ankle_y)
            ]
            
            # Calculate movement scores and add to pose data
            for name, x, y in arm_points + leg_points:
                norm_x, norm_y = self.normalize_coordinates(x, y)
                movement_score = 0.0
                
                if len(self.pose_history) > 0:
                    prev_data = self.pose_history[-1]
                    if name in prev_data:
                        movement_score = self.calculate_movement_score(
                            (norm_x, norm_y),
                            (prev_data[name]['x'], prev_data[name]['y'])
                        )
                
                pose_data[name] = {
                    'x': norm_x,
                    'y': norm_y,
                    'visibility': 1.0,
                    'movement': movement_score,
                    'raw_x': x,
                    'raw_y': y,
                    'raw_z': 0.0
                }
    

    
    def detect_hands_adaptive(self, pose_data, frame):
        """
        Adaptive hand detection using multiple methods for better results
        
        Args:
            pose_data: Dictionary containing pose data
            frame: Input color frame
        """
        # Method 1: Improved skin color detection
        self.detect_hands_and_fingers(pose_data, frame)
        
        # Method 2: If no hands detected, try motion-based detection
        if not any('hand_center' in name for name in pose_data.keys()):
            self.detect_hands_by_motion(pose_data, frame)
        
        # Method 3: If still no hands, try edge-based detection
        if not any('hand_center' in name for name in pose_data.keys()):
            self.detect_hands_by_edges(pose_data, frame)
    
    def detect_hands_by_edges(self, pose_data, frame):
        """
        Detect hands using edge detection as a fallback method
        
        Args:
            pose_data: Dictionary containing pose data
            frame: Input color frame
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Apply morphological operations to connect edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in edge image
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours for potential hands
        valid_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Edge-based contours should be medium-sized
            if area < 500 or area > 10000:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check reasonable aspect ratio
            if not (0.3 <= aspect_ratio <= 3.0):
                continue
                
            # Check position (avoid edges)
            if (x < self.frame_width * 0.05 or x > self.frame_width * 0.95 or 
                y < self.frame_height * 0.1 or y > self.frame_height * 0.9):
                continue
                
            # Check if not too close to face
            if self.is_not_near_face(contour, pose_data):
                valid_contours.append((contour, x, y, w, h))
        
        # Sort by area and process top 2
        valid_contours.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
        
        for i, (contour, x, y, w, h) in enumerate(valid_contours[:2]):
            # Determine hand side
            hand_side = 'left' if x < self.frame_width // 2 else 'right'
            
            # Hand center
            hand_center_x = x + w // 2
            hand_center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(hand_center_x, hand_center_y)
            
            # Calculate movement score
            movement_score = 0.0
            hand_center_name = f'{hand_side}_hand_center'
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if hand_center_name in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data[hand_center_name]['x'], prev_data[hand_center_name]['y'])
                    )
            
            pose_data[hand_center_name] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': hand_center_x,
                'raw_y': hand_center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h,
                'detection_method': 'edge'
            }
            
            # Estimate finger positions based on edge contour
            self.estimate_fingers_from_edges(pose_data, contour, hand_side, x, y, w, h)
    
    def estimate_fingers_from_edges(self, pose_data, contour, hand_side, x, y, w, h):
        """
        Estimate finger positions from edge contour
        
        Args:
            pose_data: Dictionary containing pose data
            contour: Edge contour
            hand_side: 'left' or 'right'
            x, y, w, h: Bounding rectangle coordinates
        """
        # Get contour points
        contour_array = np.array(contour).reshape(-1, 2)
        
        if len(contour_array) < 5:
            return
        
        # Find extreme points (likely finger tips)
        # Sort by y-coordinate to find top points
        top_points = contour_array[contour_array[:, 1].argsort()][:5]
        
        if len(top_points) >= 2:
            # Find leftmost and rightmost points among top points
            leftmost_idx = np.argmin(top_points[:, 0])
            rightmost_idx = np.argmax(top_points[:, 0])
            
            thumb_point = top_points[leftmost_idx]
            index_point = top_points[rightmost_idx]
            
            # Add finger data
            finger_points = [
                (f'{hand_side}_hand_thumb', thumb_point[0], thumb_point[1]),
                (f'{hand_side}_hand_index', index_point[0], index_point[1])
            ]
            
            for name, fx, fy in finger_points:
                norm_fx, norm_fy = self.normalize_coordinates(fx, fy)
                finger_movement = 0.0
                
                if len(self.pose_history) > 0:
                    prev_data = self.pose_history[-1]
                    if name in prev_data:
                        finger_movement = self.calculate_movement_score(
                            (norm_fx, norm_fy),
                            (prev_data[name]['x'], prev_data[name]['y'])
                        )
                
                pose_data[name] = {
                    'x': norm_fx,
                    'y': norm_fy,
                    'visibility': 1.0,
                    'movement': finger_movement,
                    'raw_x': fx,
                    'raw_y': fy,
                    'raw_z': 0.0
                }
    
    def detect_hands_and_fingers(self, pose_data, frame):
        """
        Detect hands and fingers using improved skin color detection with adaptive filtering
        
        Args:
            pose_data: Dictionary containing pose data
            frame: Input color frame
        """
        # Convert to HSV for better skin detection
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create skin mask with multiple ranges for better detection
        skin_mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        skin_mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        skin_mask3 = cv2.inRange(hsv, self.lower_skin3, self.upper_skin3)
        skin_mask4 = cv2.inRange(hsv, self.lower_skin4, self.upper_skin4)
        skin_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), 
                                  cv2.bitwise_or(skin_mask3, skin_mask4))
        
        # Apply morphological operations to clean up the mask (less aggressive)
        kernel = np.ones((5, 5), np.uint8)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
        skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply slight Gaussian blur to smooth the mask
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        
        # Find contours in the skin mask
        contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Less restrictive contour filtering for better hand detection
        valid_contours = []
        debug_info = []  # Store debug information for display
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # More restrictive area thresholds to avoid large background objects
            if area < 800 or area > 15000:  # Smaller range to avoid posters and large objects
                debug_info.append(f"Area: {area:.0f} (need 800-15000)")
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # More flexible aspect ratio for hands
            if not (0.5 <= aspect_ratio <= 2.0):  # Allow more variation
                debug_info.append(f"Aspect: {aspect_ratio:.2f} (need 0.5-2.0)")
                continue
                
            # Less restrictive position validation - allow more edge detection
            if (x < self.frame_width * 0.02 or x > self.frame_width * 0.98 or 
                y < self.frame_height * 0.05 or y > self.frame_height * 0.95):
                debug_info.append(f"Position: ({x},{y}) too close to edge")
                continue
                
            # Calculate contour properties for additional validation
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # More flexible circularity range - even more permissive
                if 0.05 <= circularity <= 0.9:  # Much more variation allowed
                    # Additional validation: check if it's not too close to face, not poster-like, and has finger-like features
                    if (self.is_not_near_face(contour, pose_data) and 
                        not self.is_poster_like(contour, hsv, x, y, w, h) and
                        self.has_finger_like_protrusions(contour, x, y, w, h)):
                        valid_contours.append(contour)
                    else:
                        # Check which filter failed
                        if not self.is_not_near_face(contour, pose_data):
                            debug_info.append("Face proximity filter failed")
                        elif self.is_poster_like(contour, hsv, x, y, w, h):
                            debug_info.append("Poster filter failed")
                        elif not self.has_finger_like_protrusions(contour, x, y, w, h):
                            debug_info.append("Finger protrusion filter failed")
                else:
                    debug_info.append(f"Circularity: {circularity:.3f} (need 0.05-0.9)")
        
        # Store debug information for display (keep only first 5)
        self.debug_filter_info = debug_info[:5]
        
        # Sort valid contours by area (largest first)
        valid_contours = sorted(valid_contours, key=cv2.contourArea, reverse=True)
        
        # Process up to 2 largest valid contours (left and right hands)
        for i, contour in enumerate(valid_contours[:2]):
            x, y, w, h = cv2.boundingRect(contour)
            
            # Determine if it's left or right hand based on position
            hand_side = 'left' if x < self.frame_width // 2 else 'right'
            
            # Hand center
            hand_center_x = x + w // 2
            hand_center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(hand_center_x, hand_center_y)
            
            # Calculate movement score
            movement_score = 0.0
            hand_center_name = f'{hand_side}_hand_center'
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if hand_center_name in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data[hand_center_name]['x'], prev_data[hand_center_name]['y'])
                    )
            
            pose_data[hand_center_name] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': hand_center_x,
                'raw_y': hand_center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h,
                'detection_method': 'skin_color'
            }
            
            # Advanced finger detection using multiple methods
            self.detect_fingers_advanced(pose_data, contour, hand_side, x, y, w, h)
    
    def is_not_near_face(self, contour, pose_data):
        """
        Check if contour is not too close to detected faces
        
        Args:
            contour: Contour to check
            pose_data: Dictionary containing pose data
            
        Returns:
            bool: True if contour is not near any detected face
        """
        if 'face_center' not in pose_data:
            return True
            
        face_center = pose_data['face_center']
        face_x = face_center['raw_x']
        face_y = face_center['raw_y']
        
        # Get contour center
        M = cv2.moments(contour)
        if M["m00"] == 0:
            return True
            
        contour_center_x = int(M["m10"] / M["m00"])
        contour_center_y = int(M["m01"] / M["m00"])
        
        # Calculate distance between face and contour
        distance = np.sqrt((face_x - contour_center_x)**2 + (face_y - contour_center_y)**2)
        
        # More flexible distance threshold - only filter if very close to face
        # This allows hands to be detected even if they're somewhat near the face
        return distance > 30  # Reduced from 50 to 30 pixels for bright lighting
    
    def is_poster_like(self, contour, hsv, x, y, w, h):
        """
        Check if contour looks like a poster (large rectangular area with uniform color)
        
        Args:
            contour: Contour to check
            hsv: HSV image for color analysis
            x, y, w, h: Bounding rectangle coordinates
            
        Returns:
            bool: True if contour looks like a poster
        """
        # Check if it's a large rectangular area
        area = cv2.contourArea(contour)
        aspect_ratio = w / h if h > 0 else 0
        
        # Posters are typically large and rectangular
        if area < 3000 or area > 40000:  # More restrictive size range
            return False
            
        if not (1.2 <= aspect_ratio <= 5.0):  # More flexible aspect ratio to catch more posters
            return False
        
        # Check color uniformity within the contour
        # Create a mask for this specific contour
        contour_mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)
        
        # Get the HSV values within this contour
        hsv_roi = hsv[y:y+h, x:x+w]
        mask_roi = contour_mask[y:y+h, x:x+w]
        
        # Calculate color statistics
        hsv_values = hsv_roi[mask_roi > 0]
        
        if len(hsv_values) == 0:
            return False
            
        # Calculate standard deviation of each channel
        h_std = np.std(hsv_values[:, 0])
        s_std = np.std(hsv_values[:, 1])
        v_std = np.std(hsv_values[:, 2])
        
        # Posters have very uniform colors (low standard deviation)
        # Hands have more color variation
        if h_std < 15 and s_std < 25 and v_std < 30:  # More aggressive uniform color detection
            return True
            
        return False
    
    def has_finger_like_protrusions(self, contour, x, y, w, h):
        """
        Check if contour has finger-like protrusions (more likely to be a hand)
        
        Args:
            contour: Contour to analyze
            x, y, w, h: Bounding rectangle coordinates
            
        Returns:
            bool: True if contour has finger-like protrusions
        """
        # Get the top portion of the contour (where fingers would be)
        contour_array = np.array(contour).reshape(-1, 2)
        top_portion = contour_array[contour_array[:, 1] < y + h * 0.7]
        
        if len(top_portion) < 10:
            return False
        
        # Check for multiple peaks in the top portion
        x_coords = top_portion[:, 0]
        y_coords = top_portion[:, 1]
        
        # Create a histogram of x-coordinates
        hist, bins = np.histogram(x_coords, bins=8)
        
        # Count peaks (local maxima)
        peak_count = 0
        for i in range(1, len(hist) - 1):
            if hist[i] > hist[i-1] and hist[i] > hist[i+1] and hist[i] > max(hist) * 0.3:
                peak_count += 1
        
        # Hands typically have 3-5 finger protrusions
        return 2 <= peak_count <= 6
    
    def detect_hands_by_motion(self, pose_data, frame):
        """
        Detect hands using motion detection as a backup method
        
        Args:
            pose_data: Dictionary containing pose data
            frame: Input color frame
        """
        # Convert to grayscale for motion detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Initialize previous frame if not exists
        if self.prev_frame is None:
            self.prev_frame = gray
            return
        
        # Calculate motion mask
        motion_mask = cv2.absdiff(self.prev_frame, gray)
        motion_mask = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)[1]
        
        # Apply morphological operations to clean up motion mask
        kernel = np.ones((5, 5), np.uint8)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours in motion mask
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter motion contours
        valid_motion_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Motion contours should be medium-sized
            if area < 800 or area > 8000:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h if h > 0 else 0
            
            # Check reasonable aspect ratio
            if not (0.5 <= aspect_ratio <= 2.0):
                continue
                
            # Check position (avoid edges)
            if (x < self.frame_width * 0.1 or x > self.frame_width * 0.9 or 
                y < self.frame_height * 0.2 or y > self.frame_height * 0.8):
                continue
                
            valid_motion_contours.append((contour, x, y, w, h))
        
        # Sort by area and process top 2
        valid_motion_contours.sort(key=lambda x: cv2.contourArea(x[0]), reverse=True)
        
        for i, (contour, x, y, w, h) in enumerate(valid_motion_contours[:2]):
            # Determine hand side
            hand_side = 'left' if x < self.frame_width // 2 else 'right'
            
            # Hand center
            hand_center_x = x + w // 2
            hand_center_y = y + h // 2
            norm_x, norm_y = self.normalize_coordinates(hand_center_x, hand_center_y)
            
            # Calculate movement score
            movement_score = 0.0
            hand_center_name = f'{hand_side}_hand_center'
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if hand_center_name in prev_data:
                    movement_score = self.calculate_movement_score(
                        (norm_x, norm_y),
                        (prev_data[hand_center_name]['x'], prev_data[hand_center_name]['y'])
                    )
            
            pose_data[hand_center_name] = {
                'x': norm_x,
                'y': norm_y,
                'visibility': 1.0,
                'movement': movement_score,
                'raw_x': hand_center_x,
                'raw_y': hand_center_y,
                'raw_z': 0.0,
                'width': w,
                'height': h,
                'detection_method': 'motion'
            }
            
            # Estimate finger positions based on motion contour
            self.estimate_fingers_from_motion(pose_data, contour, hand_side, x, y, w, h)
        
        # Update previous frame
        self.prev_frame = gray
    
    def estimate_fingers_from_motion(self, pose_data, contour, hand_side, x, y, w, h):
        """
        Estimate finger positions from motion contour
        
        Args:
            pose_data: Dictionary containing pose data
            contour: Motion contour
            hand_side: 'left' or 'right'
            x, y, w, h: Bounding rectangle coordinates
        """
        # Get contour points
        contour_array = np.array(contour).reshape(-1, 2)
        
        if len(contour_array) < 5:
            return
        
        # Find extreme points (likely finger tips)
        # Sort by y-coordinate to find top points
        top_points = contour_array[contour_array[:, 1].argsort()][:5]
        
        if len(top_points) >= 2:
            # Find leftmost and rightmost points among top points
            leftmost_idx = np.argmin(top_points[:, 0])
            rightmost_idx = np.argmax(top_points[:, 0])
            
            thumb_point = top_points[leftmost_idx]
            index_point = top_points[rightmost_idx]
            
            # Add finger data
            finger_points = [
                (f'{hand_side}_hand_thumb', thumb_point[0], thumb_point[1]),
                (f'{hand_side}_hand_index', index_point[0], index_point[1])
            ]
            
            for name, fx, fy in finger_points:
                norm_fx, norm_fy = self.normalize_coordinates(fx, fy)
                finger_movement = 0.0
                
                if len(self.pose_history) > 0:
                    prev_data = self.pose_history[-1]
                    if name in prev_data:
                        finger_movement = self.calculate_movement_score(
                            (norm_fx, norm_fy),
                            (prev_data[name]['x'], prev_data[name]['y'])
                        )
                
                pose_data[name] = {
                    'x': norm_fx,
                    'y': norm_fy,
                    'visibility': 1.0,
                    'movement': finger_movement,
                    'raw_x': fx,
                    'raw_y': fy,
                    'raw_z': 0.0
                }
    
    def detect_fingers_advanced(self, pose_data, contour, hand_side, x, y, w, h):
        """
        Advanced finger detection using multiple techniques
        
        Args:
            pose_data: Dictionary containing pose data
            contour: Hand contour
            hand_side: 'left' or 'right'
            x, y, w, h: Bounding rectangle coordinates
        """
        # Method 1: Convex hull and defects analysis
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        if defects is not None and len(defects) > 0:
            # Get convex hull points
            hull_points = cv2.convexHull(contour, returnPoints=True)
            
            if len(hull_points) >= 5:
                # Find extreme points (likely finger tips)
                hull_points = np.array(hull_points).reshape(-1, 2)
                
                # Sort by y-coordinate to find top points (finger tips)
                top_points = hull_points[hull_points[:, 1].argsort()][:5]
                
                # Find leftmost and rightmost points among top points
                leftmost_idx = np.argmin(top_points[:, 0])
                rightmost_idx = np.argmax(top_points[:, 0])
                
                thumb_point = top_points[leftmost_idx]
                index_point = top_points[rightmost_idx]
                
                # Add finger data
                finger_points = [
                    (f'{hand_side}_hand_thumb', thumb_point[0], thumb_point[1]),
                    (f'{hand_side}_hand_index', index_point[0], index_point[1])
                ]
                
                for name, fx, fy in finger_points:
                    norm_fx, norm_fy = self.normalize_coordinates(fx, fy)
                    finger_movement = 0.0
                    
                    if len(self.pose_history) > 0:
                        prev_data = self.pose_history[-1]
                        if name in prev_data:
                            finger_movement = self.calculate_movement_score(
                                (norm_fx, norm_fy),
                                (prev_data[name]['x'], prev_data[name]['y'])
                            )
                    
                    pose_data[name] = {
                        'x': norm_fx,
                        'y': norm_fy,
                        'visibility': 1.0,
                        'movement': finger_movement,
                        'raw_x': fx,
                        'raw_y': fy,
                        'raw_z': 0.0
                    }
                return
        
        # Method 2: Contour analysis for finger-like protrusions
        # Find the top portion of the hand contour
        contour_array = np.array(contour).reshape(-1, 2)
        top_portion = contour_array[contour_array[:, 1] < y + h * 0.6]
        
        if len(top_portion) > 0:
            # Find local maxima in the top portion
            try:
                from scipy.signal import find_peaks
                
                # Create a histogram of x-coordinates
                x_coords = top_portion[:, 0]
                hist, bins = np.histogram(x_coords, bins=10)
                
                # Find peaks in the histogram
                peaks, _ = find_peaks(hist, height=max(hist) * 0.3)
            except ImportError:
                # Fallback if scipy is not available
                peaks = []
            
            if len(peaks) >= 2:
                # Use the leftmost and rightmost peaks
                left_peak_x = bins[peaks[0]]
                right_peak_x = bins[peaks[-1]]
                
                # Find corresponding y-coordinates
                left_points = top_portion[top_portion[:, 0] == left_peak_x]
                right_points = top_portion[top_portion[:, 0] == right_peak_x]
                
                if len(left_points) > 0 and len(right_points) > 0:
                    thumb_x, thumb_y = left_points[0]
                    index_x, index_y = right_points[0]
                    
                    # Add finger data
                    finger_points = [
                        (f'{hand_side}_hand_thumb', thumb_x, thumb_y),
                        (f'{hand_side}_hand_index', index_x, index_y)
                    ]
                    
                    for name, fx, fy in finger_points:
                        norm_fx, norm_fy = self.normalize_coordinates(fx, fy)
                        finger_movement = 0.0
                        
                        if len(self.pose_history) > 0:
                            prev_data = self.pose_history[-1]
                            if name in prev_data:
                                finger_movement = self.calculate_movement_score(
                                    (norm_fx, norm_fy),
                                    (prev_data[name]['x'], prev_data[name]['y'])
                                )
                        
                        pose_data[name] = {
                            'x': norm_fx,
                            'y': norm_fy,
                            'visibility': 1.0,
                            'movement': finger_movement,
                            'raw_x': fx,
                            'raw_y': fy,
                            'raw_z': 0.0
                        }
                    return
        
        # Method 3: Fallback - estimate based on hand bounding box
        thumb_x = x + w // 4
        thumb_y = y + h // 4
        index_x = x + 3 * w // 4
        index_y = y + h // 4
        
        # Add finger data
        finger_points = [
            (f'{hand_side}_hand_thumb', thumb_x, thumb_y),
            (f'{hand_side}_hand_index', index_x, index_y)
        ]
        
        for name, fx, fy in finger_points:
            norm_fx, norm_fy = self.normalize_coordinates(fx, fy)
            finger_movement = 0.0
            
            if len(self.pose_history) > 0:
                prev_data = self.pose_history[-1]
                if name in prev_data:
                    finger_movement = self.calculate_movement_score(
                        (norm_fx, norm_fy),
                        (prev_data[name]['x'], prev_data[name]['y'])
                    )
            
            pose_data[name] = {
                'x': norm_fx,
                'y': norm_fy,
                'visibility': 1.0,
                'movement': finger_movement,
                'raw_x': fx,
                'raw_y': fy,
                'raw_z': 0.0
            }
    
    def calculate_pose_metrics(self, pose_data):
        """
        Calculate overall pose metrics and body part relationships
        
        Args:
            pose_data: Dictionary containing pose data
            
        Returns:
            dict: Overall pose metrics
        """
        metrics = {}
        
        if not pose_data:
            return metrics
        
        # Calculate face symmetry (0-1, where 1 is perfectly symmetric)
        if 'face_left_eye' in pose_data and 'face_right_eye' in pose_data:
            left_y = pose_data['face_left_eye']['y']
            right_y = pose_data['face_right_eye']['y']
            symmetry_score = 1.0 - abs(left_y - right_y)
            metrics['face_symmetry'] = max(0.0, symmetry_score)
        
        # Calculate upper body symmetry
        if 'upper_body_left' in pose_data and 'upper_body_right' in pose_data:
            left_y = pose_data['upper_body_left']['y']
            right_y = pose_data['upper_body_right']['y']
            symmetry_score = 1.0 - abs(left_y - right_y)
            metrics['upper_body_symmetry'] = max(0.0, symmetry_score)
        
        # Calculate full body symmetry
        if 'full_body_left' in pose_data and 'full_body_right' in pose_data:
            left_y = pose_data['full_body_left']['y']
            right_y = pose_data['full_body_right']['y']
            symmetry_score = 1.0 - abs(left_y - right_y)
            metrics['full_body_symmetry'] = max(0.0, symmetry_score)
        
        # Calculate face position relative to frame center
        if 'face_center' in pose_data:
            face_x = pose_data['face_center']['x']
            face_y = pose_data['face_center']['y']
            center_distance = np.sqrt((face_x - 0.5)**2 + (face_y - 0.5)**2)
            metrics['face_center_distance'] = min(center_distance, 1.0)
        
        # Calculate overall movement score
        total_movement = sum(data['movement'] for data in pose_data.values())
        avg_movement = total_movement / len(pose_data) if pose_data else 0.0
        metrics['overall_movement'] = avg_movement
        
        # Calculate pose stability (inverse of movement)
        metrics['pose_stability'] = 1.0 - avg_movement
        
        # Calculate arm angles
        if all(k in pose_data for k in ['left_arm_shoulder', 'left_arm_elbow', 'left_arm_wrist']):
            left_arm_angle = self.calculate_angle(
                pose_data['left_arm_shoulder']['x'], pose_data['left_arm_shoulder']['y'],
                pose_data['left_arm_elbow']['x'], pose_data['left_arm_elbow']['y'],
                pose_data['left_arm_wrist']['x'], pose_data['left_arm_wrist']['y']
            )
            metrics['left_arm_angle'] = left_arm_angle / 180.0  # Normalize to 0-1
        
        if all(k in pose_data for k in ['right_arm_shoulder', 'right_arm_elbow', 'right_arm_wrist']):
            right_arm_angle = self.calculate_angle(
                pose_data['right_arm_shoulder']['x'], pose_data['right_arm_shoulder']['y'],
                pose_data['right_arm_elbow']['x'], pose_data['right_arm_elbow']['y'],
                pose_data['right_arm_wrist']['x'], pose_data['right_arm_wrist']['y']
            )
            metrics['right_arm_angle'] = right_arm_angle / 180.0  # Normalize to 0-1
        
        # Calculate leg angles
        if all(k in pose_data for k in ['left_leg_hip', 'left_leg_knee', 'left_leg_ankle']):
            left_leg_angle = self.calculate_angle(
                pose_data['left_leg_hip']['x'], pose_data['left_leg_hip']['y'],
                pose_data['left_leg_knee']['x'], pose_data['left_leg_knee']['y'],
                pose_data['left_leg_ankle']['x'], pose_data['left_leg_ankle']['y']
            )
            metrics['left_leg_angle'] = left_leg_angle / 180.0  # Normalize to 0-1
        
        if all(k in pose_data for k in ['right_leg_hip', 'right_leg_knee', 'right_leg_ankle']):
            right_leg_angle = self.calculate_angle(
                pose_data['right_leg_hip']['x'], pose_data['right_leg_hip']['y'],
                pose_data['right_leg_knee']['x'], pose_data['right_leg_knee']['y'],
                pose_data['right_leg_ankle']['x'], pose_data['right_leg_ankle']['y']
            )
            metrics['right_leg_angle'] = right_leg_angle / 180.0  # Normalize to 0-1
        
        # Calculate hand movement
        hand_movement = 0.0
        hand_count = 0
        for hand_name in ['left_hand_center', 'right_hand_center']:
            if hand_name in pose_data:
                hand_movement += pose_data[hand_name]['movement']
                hand_count += 1
        
        if hand_count > 0:
            metrics['hand_movement'] = hand_movement / hand_count
        
        # Calculate detection confidence (number of detected features)
        metrics['detection_confidence'] = len(pose_data) / 28.0  # Normalize by max possible features (28 total)
        
        return metrics
    
    def calculate_angle(self, x1, y1, x2, y2, x3, y3):
        """
        Calculate angle between three points
        
        Args:
            x1, y1, x2, y2, x3, y3: Coordinates of three points
            
        Returns:
            float: Angle in degrees
        """
        a = np.array([x1, y1])
        b = np.array([x2, y2])
        c = np.array([x3, y3])
        
        ba = a - b
        bc = c - b
        
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
        
        return np.degrees(angle)
    
    def draw_detections(self, image, pose_data):
        """
        Draw detection rectangles on the image
        
        Args:
            image: Input image
            pose_data: Dictionary containing pose data
        """
        # Only draw if we have pose data (optimization)
        if not pose_data:
            return
            
        # Draw rectangles based on pose data instead of re-detecting
        for name, data in pose_data.items():
            if 'width' in data and 'height' in data:
                x = int(data['raw_x'] - data['width'] // 2)
                y = int(data['raw_y'] - data['height'] // 2)
                w = int(data['width'])
                h = int(data['height'])
                
                if 'face' in name:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    cv2.putText(image, 'Face', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
                elif 'upper_body' in name:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, 'Upper Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                elif 'full_body' in name:
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(image, 'Full Body', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                elif 'hand_center' in name:
                    # Use different colors based on detection method
                    if 'detection_method' in data and data['detection_method'] == 'skin_color':
                        color = (0, 255, 255)  # Yellow for skin color detection
                        label = 'Hand (Skin)'
                    elif 'detection_method' in data and data['detection_method'] == 'motion':
                        color = (0, 255, 0)    # Green for motion detection
                        label = 'Hand (Motion)'
                    elif 'detection_method' in data and data['detection_method'] == 'edge':
                        color = (255, 0, 255)  # Magenta for edge detection
                        label = 'Hand (Edge)'
                    else:
                        color = (0, 255, 255)  # Default yellow
                        label = 'Hand'
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    def draw_pose_data(self, image, pose_data, metrics):
        """
        Draw pose landmarks and metrics on the image
        
        Args:
            image: Input image
            pose_data: Dictionary containing pose data
            metrics: Dictionary containing pose metrics
        """
        # Draw pose landmarks with different colors for different body parts
        if pose_data:
            # Draw body parts with different colors
            body_part_colors = {
                'face': (255, 0, 0),      # Blue
                'upper_body': (0, 255, 0), # Green
                'full_body': (0, 0, 255),  # Red
                'arm': (255, 255, 0),      # Cyan
                'leg': (255, 0, 255),      # Magenta
                'hand': (0, 255, 255)      # Yellow
            }
            
            for name, data in pose_data.items():
                x = int(data['raw_x'])
                y = int(data['raw_y'])
                
                # Determine body part type and color
                part_type = None
                if 'face' in name:
                    part_type = 'face'
                elif 'upper_body' in name or 'full_body' in name:
                    part_type = 'upper_body' if 'upper_body' in name else 'full_body'
                elif 'arm' in name:
                    part_type = 'arm'
                elif 'leg' in name:
                    part_type = 'leg'
                elif 'hand' in name:
                    part_type = 'hand'
                
                if part_type:
                    base_color = body_part_colors[part_type]
                    movement = data['movement']
                    
                    # Blend color with movement (red = high movement)
                    color = (
                        int(base_color[0] * (1 - movement) + 255 * movement),
                        int(base_color[1] * (1 - movement)),
                        int(base_color[2] * (1 - movement))
                    )
                    
                    # Draw circle for landmark
                    radius = 3 if 'center' in name else 2
                    cv2.circle(image, (x, y), radius, color, -1)
                    
                    # Draw connections for arms and legs
                    if 'shoulder' in name or 'hip' in name:
                        # Draw arm/leg connections
                        if 'left_arm_shoulder' in name and 'left_arm_elbow' in pose_data:
                            elbow_x = int(pose_data['left_arm_elbow']['raw_x'])
                            elbow_y = int(pose_data['left_arm_elbow']['raw_y'])
                            cv2.line(image, (x, y), (elbow_x, elbow_y), color, 2)
                        
                        if 'right_arm_shoulder' in name and 'right_arm_elbow' in pose_data:
                            elbow_x = int(pose_data['right_arm_elbow']['raw_x'])
                            elbow_y = int(pose_data['right_arm_elbow']['raw_y'])
                            cv2.line(image, (x, y), (elbow_x, elbow_y), color, 2)
                        
                        if 'left_leg_hip' in name and 'left_leg_knee' in pose_data:
                            knee_x = int(pose_data['left_leg_knee']['raw_x'])
                            knee_y = int(pose_data['left_leg_knee']['raw_y'])
                            cv2.line(image, (x, y), (knee_x, knee_y), color, 2)
                        
                        if 'right_leg_hip' in name and 'right_leg_knee' in pose_data:
                            knee_x = int(pose_data['right_leg_knee']['raw_x'])
                            knee_y = int(pose_data['right_leg_knee']['raw_y'])
                            cv2.line(image, (x, y), (knee_x, knee_y), color, 2)
                    
                    # Draw simplified text for key points
                    if 'center' in name or 'wrist' in name or 'ankle' in name:
                        text = f"{name.split('_')[0]}: m={data['movement']:.2f}"
                        cv2.putText(image, text, (x + 5, y - 5), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1)
        
        # Draw enhanced metrics
        y_offset = 30
        key_metrics = ['overall_movement', 'pose_stability', 'detection_confidence', 
                      'hand_movement', 'left_arm_angle', 'right_arm_angle']
        for metric_name in key_metrics:
            if metric_name in metrics:
                value = metrics[metric_name]
                text = f"{metric_name}: {value:.3f}"
                cv2.putText(image, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2)
                y_offset += 18
    
    def show_debug_info(self, frame, pose_data):
        """
        Show debug information including skin detection masks
        
        Args:
            frame: Input frame
            pose_data: Pose data dictionary
        """
        # Create a smaller debug window
        debug_frame = frame.copy()
        
        # Convert to HSV and create skin mask for visualization
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        skin_mask1 = cv2.inRange(hsv, self.lower_skin, self.upper_skin)
        skin_mask2 = cv2.inRange(hsv, self.lower_skin2, self.upper_skin2)
        skin_mask3 = cv2.inRange(hsv, self.lower_skin3, self.upper_skin3)
        skin_mask4 = cv2.inRange(hsv, self.lower_skin4, self.upper_skin4)
        skin_mask = cv2.bitwise_or(cv2.bitwise_or(skin_mask1, skin_mask2), 
                                  cv2.bitwise_or(skin_mask3, skin_mask4))
        
        # Convert mask to BGR for display
        skin_mask_bgr = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        
        # Resize for display
        height, width = debug_frame.shape[:2]
        debug_width = width // 2  # Make debug windows larger
        debug_height = height // 2
        
        skin_mask_resized = cv2.resize(skin_mask_bgr, (debug_width, debug_height))
        
        # Place skin mask in top-right corner
        debug_frame[0:debug_height, width-debug_width:width] = skin_mask_resized
        
        # Add text overlay with better visibility
        cv2.putText(debug_frame, "SKIN MASK", (width-debug_width+5, 25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Add current skin detection parameters
        cv2.putText(debug_frame, f"S: {self.lower_skin[1]}-{self.upper_skin[1]}", (width-debug_width+5, 45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        cv2.putText(debug_frame, f"V: {self.lower_skin[2]}-{self.upper_skin[2]}", (width-debug_width+5, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        
        # Show hand detection info with method
        hand_count = sum(1 for name in pose_data.keys() if 'hand_center' in name)
        skin_hands = sum(1 for name, data in pose_data.items() 
                        if 'hand_center' in name and 'detection_method' in data and data['detection_method'] == 'skin_color')
        motion_hands = sum(1 for name, data in pose_data.items() 
                          if 'hand_center' in name and 'detection_method' in data and data['detection_method'] == 'motion')
        edge_hands = sum(1 for name, data in pose_data.items() 
                        if 'hand_center' in name and 'detection_method' in data and data['detection_method'] == 'edge')
        other_hands = hand_count - skin_hands - motion_hands - edge_hands
        
        cv2.putText(debug_frame, f"Hands: {hand_count} (Skin: {skin_hands}, Motion: {motion_hands}, Edge: {edge_hands}, Other: {other_hands})", (10, height-40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Show total detections
        cv2.putText(debug_frame, f"Total Features: {len(pose_data)}", (10, height-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add calibration instructions
        cv2.putText(debug_frame, "Press +/- to adjust skin detection", (10, height-60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(debug_frame, "Press 'd' to toggle debug mode", (10, height-45), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Show motion detection info
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self.prev_frame is not None:
            motion_mask = cv2.absdiff(self.prev_frame, gray)
            motion_mask = cv2.threshold(motion_mask, 25, 255, cv2.THRESH_BINARY)[1]
        else:
            motion_mask = np.zeros_like(gray)
        
        # Convert motion mask to BGR for display
        motion_mask_bgr = cv2.cvtColor(motion_mask, cv2.COLOR_GRAY2BGR)
        motion_mask_resized = cv2.resize(motion_mask_bgr, (debug_width, debug_height))
        
        # Place motion mask in bottom-right corner
        debug_frame[height-debug_height:height, width-debug_width:width] = motion_mask_resized
        
        # Add text overlay for motion mask with better visibility
        cv2.putText(debug_frame, "MOTION MASK", (width-debug_width+5, height-debug_height+25), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Update prev_frame for next iteration
        self.prev_frame = gray
        
        # Show filter debug information if available
        if hasattr(self, 'debug_filter_info') and self.debug_filter_info:
            y_offset = 80
            cv2.putText(debug_frame, "Filter Debug:", (10, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 255), 1)
            y_offset += 15
            for i, info in enumerate(self.debug_filter_info[:3]):  # Show first 3 filter issues
                cv2.putText(debug_frame, info, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
                y_offset += 12
        
        return debug_frame
    
    def run(self):
        """
        Main loop to capture video and process pose detection
        """
        cap = cv2.VideoCapture(0)
        
        # Set camera properties for speed
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        cap.set(cv2.CAP_PROP_FPS, 30)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size for lower latency
        
        print("Fast Human Pose Tracker Started!")
        print("Press 'q' to quit, 's' to save current pose data, 'd' to toggle debug mode")
        print("Press '+'/'-' to adjust skin detection sensitivity, 'c' for calibration mode")
        print(f"Resolution: {self.frame_width}x{self.frame_height}, Frame skip: {self.frame_skip}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
            
            # Flip the frame horizontally for a later selfie-view display
            frame = cv2.flip(frame, 1)
            
            # Extract pose data using OpenCV detection
            pose_data = self.extract_pose_data(frame)
            
            # Add to history
            if pose_data:
                self.pose_history.append(pose_data)
            
            # Calculate overall metrics
            metrics = self.calculate_pose_metrics(pose_data)
            
            # Draw detection rectangles and pose data
            self.draw_detections(frame, pose_data)
            self.draw_pose_data(frame, pose_data, metrics)
            
            # Show debug information if enabled
            if self.debug_mode:
                debug_frame = self.show_debug_info(frame, pose_data)
                # Display the debug frame
                cv2.imshow('Fast Human Pose Tracker', debug_frame)
            else:
                # Display the original frame
                cv2.imshow('Fast Human Pose Tracker', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s') and pose_data:
                self.save_pose_data(pose_data, metrics)
            elif key == ord('d'):  # Toggle debug mode
                self.debug_mode = not self.debug_mode
                print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            elif key == ord('c'):  # Toggle calibration mode
                self.calibration_mode = not getattr(self, 'calibration_mode', False)
                print(f"Calibration mode: {'ON' if self.calibration_mode else 'OFF'}")
                if self.calibration_mode:
                    print("In calibration mode: Use arrow keys to adjust skin detection")
                    print("Up/Down: Adjust saturation, Left/Right: Adjust value")
            elif key == ord('+') or key == ord('='):  # Increase skin detection sensitivity
                self.adjust_skin_detection(1)
            elif key == ord('-') or key == ord('_'):  # Decrease skin detection sensitivity
                self.adjust_skin_detection(-1)
        
        cap.release()
        cv2.destroyAllWindows()
    
    def save_pose_data(self, pose_data, metrics):
        """
        Save current pose data to a file
        
        Args:
            pose_data: Dictionary containing pose data
            metrics: Dictionary containing pose metrics
        """
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"pose_data_{timestamp}.txt"
        
        with open(filename, 'w') as f:
            f.write("Human Pose Data\n")
            f.write("=" * 50 + "\n")
            f.write(f"Timestamp: {timestamp}\n\n")
            
            f.write("Body Part Positions (Normalized 0-1):\n")
            f.write("-" * 40 + "\n")
            for name, data in pose_data.items():
                f.write(f"{name}:\n")
                f.write(f"  Position: ({data['x']:.3f}, {data['y']:.3f})\n")
                f.write(f"  Movement: {data['movement']:.3f}\n")
                f.write(f"  Visibility: {data['visibility']:.3f}\n")
                f.write(f"  3D Position: ({data['raw_x']:.3f}, {data['raw_y']:.3f}, {data['raw_z']:.3f})\n\n")
            
            f.write("Overall Metrics:\n")
            f.write("-" * 20 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name}: {value:.3f}\n")
        
        print(f"Pose data saved to {filename}")
    
    def adjust_skin_detection(self, direction):
        """
        Adjust skin detection parameters for better hand detection
        
        Args:
            direction: 1 for more sensitive, -1 for less sensitive
        """
        adjustment = 5 * direction
        
        # Adjust primary skin range
        self.lower_skin[1] = max(0, min(255, int(self.lower_skin[1]) - adjustment))  # Saturation
        self.upper_skin[1] = max(0, min(255, int(self.upper_skin[1]) + adjustment))  # Saturation
        self.lower_skin[2] = max(0, min(255, int(self.lower_skin[2]) - adjustment))  # Value
        self.upper_skin[2] = max(0, min(255, int(self.upper_skin[2]) + adjustment))  # Value
        
        # Adjust alternative ranges
        self.lower_skin2[1] = max(0, min(255, int(self.lower_skin2[1]) - adjustment))
        self.upper_skin2[1] = max(0, min(255, int(self.upper_skin2[1]) + adjustment))
        self.lower_skin2[2] = max(0, min(255, int(self.lower_skin2[2]) - adjustment))
        self.upper_skin2[2] = max(0, min(255, int(self.upper_skin2[2]) + adjustment))
        
        self.lower_skin3[1] = max(0, min(255, int(self.lower_skin3[1]) - adjustment))
        self.upper_skin3[1] = max(0, min(255, int(self.upper_skin3[1]) + adjustment))
        self.lower_skin3[2] = max(0, min(255, int(self.lower_skin3[2]) - adjustment))
        self.upper_skin3[2] = max(0, min(255, int(self.upper_skin3[2]) + adjustment))
        
        self.lower_skin4[1] = max(0, min(255, int(self.lower_skin4[1]) - adjustment))
        self.upper_skin4[1] = max(0, min(255, int(self.upper_skin4[1]) + adjustment))
        self.lower_skin4[2] = max(0, min(255, int(self.lower_skin4[2]) - adjustment))
        self.upper_skin4[2] = max(0, min(255, int(self.upper_skin4[2]) + adjustment))
        
        print(f"Skin detection {'increased' if direction > 0 else 'decreased'} sensitivity")
        print(f"Primary range: S[{self.lower_skin[1]}-{self.upper_skin[1]}], V[{self.lower_skin[2]}-{self.upper_skin[2]}]")

def main():
    """
    Main function to run the human pose tracker
    """
    try:
        tracker = HumanPoseTracker()
        tracker.run()
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
