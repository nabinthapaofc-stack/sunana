"""
Hand Sign Recognizer - Real-time ASL alphabet and number recognition
Uses MediaPipe Hands for landmark detection and rule-based classification
"""

import cv2
import numpy as np
import mediapipe as mp
import json
from typing import Optional, Dict, Tuple, List
import config


class HandSignRecognizer:
    """
    Real-time ASL alphabet and number recognition using MediaPipe Hands.
    
    Uses rule-based classification based on finger positions and angles
    to recognize standard ASL finger spelling (A-Z) and numbers (0-9).
    
    Usage:
        recognizer = HandSignRecognizer()
        result = recognizer.recognize(frame)
        print(result["character"])  # e.g., "A"
    """
    
    def __init__(self):
        """Initialize the recognizer with MediaPipe Hands and load mappings."""
        try:
            # Initialize MediaPipe Hands
            self.mp_hands = mp.solutions.hands
            self.mp_drawing = mp.solutions.drawing_utils
            self.mp_drawing_styles = mp.solutions.drawing_styles
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=config.MAX_NUM_HANDS,
                min_detection_confidence=config.MIN_DETECTION_CONFIDENCE,
                min_tracking_confidence=config.MIN_TRACKING_CONFIDENCE,
                model_complexity=config.MODEL_COMPLEXITY
            )
            self.mediapipe_available = True
            print("✓ Hand Sign Recognizer initialized")
        except Exception as e:
            print(f"! WARNING: MediaPipe could not be initialized: {e}")
            print("! Running in SAFE MODE (Recognition disabled)")
            print("! Please run with Python 3.8-3.11 for gesture recognition.")
            self.mediapipe_available = False
            self.mp_hands = None
            self.hands = None
        
        # Load gesture mappings
        self.mappings = self._load_mappings()
        
        # Stability tracking
        self.last_gesture = None
        self.gesture_count = 0
        self.confirmed_gesture = None
        
        # Landmark indices for reference
        self.FINGER_TIPS = [4, 8, 12, 16, 20]
        self.FINGER_PIPS = [3, 6, 10, 14, 18]
        self.FINGER_MCPS = [2, 5, 9, 13, 17]
        
        print("✓ Hand Sign Recognizer initialized")
    
    def _load_mappings(self) -> Dict:
        """Load character mappings from JSON file."""
        try:
            with open(config.GESTURE_MAPPINGS_PATH, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Warning: Mappings file not found at {config.GESTURE_MAPPINGS_PATH}")
            return {"alphabets": {}, "numbers": {}, "nepali": {}}
    
    def _get_landmark_coords(self, landmarks, idx: int) -> Tuple[float, float, float]:
        """Get x, y, z coordinates for a landmark index."""
        lm = landmarks.landmark[idx]
        return lm.x, lm.y, lm.z
    
    def _get_finger_states(self, landmarks, handedness: str) -> Dict[str, bool]:
        """
        Determine which fingers are extended.
        
        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"
            
        Returns:
            Dict with keys: thumb, index, middle, ring, pinky
            Values: True if extended, False if folded
        """
        states = {}
        
        # Get landmark positions
        wrist = self._get_landmark_coords(landmarks, 0)
        
        # Thumb: Check if thumb tip is to the side of thumb IP
        # Thumb is special - check horizontal position relative to palm
        thumb_tip = self._get_landmark_coords(landmarks, 4)
        thumb_ip = self._get_landmark_coords(landmarks, 3)
        thumb_mcp = self._get_landmark_coords(landmarks, 2)
        index_mcp = self._get_landmark_coords(landmarks, 5)
        
        # For right hand, thumb extended means tip is to the left of IP
        # For left hand, it's the opposite
        if handedness == "Right":
            states["thumb"] = thumb_tip[0] < thumb_ip[0]
        else:
            states["thumb"] = thumb_tip[0] > thumb_ip[0]
        
        # Other fingers: Check if fingertip is above PIP joint (lower y = higher position)
        finger_names = ["index", "middle", "ring", "pinky"]
        for i, name in enumerate(finger_names):
            tip_idx = self.FINGER_TIPS[i + 1]  # +1 because thumb is at 0
            pip_idx = self.FINGER_PIPS[i + 1]
            
            tip = self._get_landmark_coords(landmarks, tip_idx)
            pip = self._get_landmark_coords(landmarks, pip_idx)
            
            # Finger is extended if tip is above (lower y) the PIP joint
            states[name] = tip[1] < pip[1]
        
        return states
    
    def _count_extended_fingers(self, finger_states: Dict[str, bool]) -> int:
        """Count how many fingers are extended."""
        return sum(1 for v in finger_states.values() if v)
    
    def _get_fingertip_distances(self, landmarks) -> Dict[str, float]:
        """Calculate distances between fingertips and thumb tip."""
        thumb_tip = self._get_landmark_coords(landmarks, 4)
        
        distances = {}
        finger_names = ["index", "middle", "ring", "pinky"]
        
        for i, name in enumerate(finger_names):
            tip_idx = self.FINGER_TIPS[i + 1]
            tip = self._get_landmark_coords(landmarks, tip_idx)
            
            # Euclidean distance
            dist = np.sqrt(
                (tip[0] - thumb_tip[0])**2 + 
                (tip[1] - thumb_tip[1])**2 + 
                (tip[2] - thumb_tip[2])**2
            )
            distances[name] = dist
        
        return distances
    
    def _classify_gesture(self, landmarks, handedness: str) -> Tuple[Optional[str], float]:
        """
        Classify the hand gesture using rule-based logic.
        
        Implements complete ASL alphabet (A-Z) and numbers (0-9) recognition
        with conflict resolution for similar-looking signs.
        
        Args:
            landmarks: MediaPipe hand landmarks
            handedness: "Left" or "Right"
            
        Returns:
            Tuple of (gesture_name, confidence)
        """
        finger_states = self._get_finger_states(landmarks, handedness)
        extended_count = self._count_extended_fingers(finger_states)
        thumb_distances = self._get_fingertip_distances(landmarks)
        
        # Get key landmarks for detailed analysis
        thumb_tip = self._get_landmark_coords(landmarks, 4)
        thumb_ip = self._get_landmark_coords(landmarks, 3)
        thumb_mcp = self._get_landmark_coords(landmarks, 2)
        
        index_tip = self._get_landmark_coords(landmarks, 8)
        index_dip = self._get_landmark_coords(landmarks, 7)
        index_pip = self._get_landmark_coords(landmarks, 6)
        index_mcp = self._get_landmark_coords(landmarks, 5)
        
        middle_tip = self._get_landmark_coords(landmarks, 12)
        middle_dip = self._get_landmark_coords(landmarks, 11)
        middle_pip = self._get_landmark_coords(landmarks, 10)
        middle_mcp = self._get_landmark_coords(landmarks, 9)
        
        ring_tip = self._get_landmark_coords(landmarks, 16)
        ring_pip = self._get_landmark_coords(landmarks, 14)
        ring_mcp = self._get_landmark_coords(landmarks, 13)
        
        pinky_tip = self._get_landmark_coords(landmarks, 20)
        pinky_pip = self._get_landmark_coords(landmarks, 18)
        pinky_mcp = self._get_landmark_coords(landmarks, 17)
        
        wrist = self._get_landmark_coords(landmarks, 0)
        
        # Helper: check if fingers are horizontal (pointing sideways)
        def is_horizontal(tip, pip):
            """Check if finger is more horizontal than vertical."""
            return abs(tip[0] - pip[0]) > abs(tip[1] - pip[1])
        
        # Helper: check if hand is pointing down
        def is_pointing_down(tip, mcp):
            """Check if fingertip is below MCP (pointing down)."""
            return tip[1] > mcp[1] + 0.05
        
        # Helper: distance between two points
        def dist(p1, p2):
            return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)
        
        # Helper: check if index is bent (hook shape for X)
        def is_index_hooked():
            # Index tip should be close to or below DIP, but DIP above PIP
            return (index_tip[1] > index_dip[1] - 0.02 and 
                    index_dip[1] < index_pip[1] and
                    not finger_states["index"])
        
        # Helper: check if fingers are crossed (for R)
        def are_index_middle_crossed():
            # When crossed, middle tip x is on same side as index tip
            if handedness == "Right":
                return middle_tip[0] < index_tip[0] and finger_states["index"] and finger_states["middle"]
            else:
                return middle_tip[0] > index_tip[0] and finger_states["index"] and finger_states["middle"]
        
        # ========== PRIORITY CHECKS (Most distinctive first) ==========
        
        # Y - Thumb and pinky extended, others folded (hang loose)
        if (finger_states["thumb"] and 
            not finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            finger_states["pinky"]):
            return ("Y", 0.92)
        
        # I - Only pinky extended
        if (not finger_states["thumb"] and
            not finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            finger_states["pinky"]):
            return ("I", 0.92)
        
        # L - Index and thumb form L shape
        if (finger_states["thumb"] and 
            finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            # Check for L shape (thumb perpendicular to index)
            if abs(thumb_tip[0] - index_tip[0]) > 0.08:
                return ("L", 0.90)
        
        # 5 - All five fingers extended (open palm)
        if all(finger_states.values()):
            return ("5", 0.95)
        
        # W - Index, middle, ring extended and spread, pinky folded
        if (finger_states["index"] and 
            finger_states["middle"] and 
            finger_states["ring"] and 
            not finger_states["pinky"] and
            not finger_states["thumb"]):
            return ("W", 0.88)
        
        # 3 - Thumb, index, middle extended
        if (finger_states["thumb"] and 
            finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            return ("3", 0.88)
        
        # 4/B - Four fingers up, thumb folded
        if (not finger_states["thumb"] and 
            finger_states["index"] and 
            finger_states["middle"] and 
            finger_states["ring"] and 
            finger_states["pinky"]):
            # Check if fingers are together (B) or spread (4)
            finger_spread = abs(index_tip[0] - pinky_tip[0])
            if finger_spread < 0.12:
                return ("B", 0.88)
            else:
                return ("4", 0.85)
        
        # ========== HORIZONTAL GESTURES (G, H, P, Q) ==========
        
        # G - Index extended horizontally, thumb up
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            if is_horizontal(index_tip, index_pip) and not is_pointing_down(index_tip, index_mcp):
                return ("G", 0.85)
        
        # H - Index and middle extended horizontally
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            if is_horizontal(index_tip, index_pip) and is_horizontal(middle_tip, middle_pip):
                return ("H", 0.85)
        
        # P - Like K but pointing down (index and middle down, thumb out)
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            if is_pointing_down(index_tip, index_mcp) and is_pointing_down(middle_tip, middle_mcp):
                return ("P", 0.82)
        
        # Q - Like G but pointing down
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            if is_pointing_down(index_tip, index_mcp):
                return ("Q", 0.82)
        
        # ========== TWO FINGER GESTURES (U, V, K, R) ==========
        
        # R - Index and middle crossed
        if are_index_middle_crossed() and not finger_states["ring"] and not finger_states["pinky"]:
            return ("R", 0.85)
        
        # K - Index and middle up spread, thumb between them
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            # Check thumb is between index and middle
            thumb_between = (min(index_tip[0], middle_tip[0]) < thumb_tip[0] < max(index_tip[0], middle_tip[0]))
            spread = abs(index_tip[0] - middle_tip[0])
            if thumb_between and spread > 0.03 and finger_states["thumb"]:
                return ("K", 0.85)
        
        # V/2 - Index and middle spread (peace sign)
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            spread = abs(index_tip[0] - middle_tip[0])
            if spread > 0.05:
                return ("V", 0.90)
        
        # U - Index and middle together pointing up
        if (finger_states["index"] and 
            finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            spread = abs(index_tip[0] - middle_tip[0])
            if spread < 0.04:
                return ("U", 0.88)
        
        # ========== SINGLE EXTENDED FINGER (1, D) ==========
        
        # D - Index up, thumb touches middle finger
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"]):
            if thumb_distances["middle"] < 0.07:
                return ("D", 0.85)
        
        # 1 - Only index extended, thumb folded
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"] and
            not finger_states["thumb"]):
            return ("1", 0.92)
        
        # ========== THUMB-FINGER TOUCHING (F, 6, 7, 8, 9) ==========
        
        # F/9 - Index and thumb form circle, other 3 fingers up
        if (finger_states["middle"] and 
            finger_states["ring"] and 
            finger_states["pinky"] and
            thumb_distances["index"] < 0.06):
            return ("F", 0.88)
        
        # 6 - Thumb touches pinky, other fingers up
        if (finger_states["index"] and 
            finger_states["middle"] and 
            finger_states["ring"] and 
            dist(thumb_tip, pinky_tip) < 0.06):
            return ("6", 0.85)
        
        # 7 - Thumb touches ring finger
        if (finger_states["index"] and 
            finger_states["middle"] and 
            finger_states["pinky"] and 
            dist(thumb_tip, ring_tip) < 0.06):
            return ("7", 0.85)
        
        # 8 - Thumb touches middle finger
        if (finger_states["index"] and 
            finger_states["ring"] and 
            finger_states["pinky"] and 
            dist(thumb_tip, middle_tip) < 0.06):
            return ("8", 0.85)
        
        # ========== CURVED/CIRCLE SHAPES (C, O, 0) ==========
        
        # O/0 - All fingers curve to form circle with thumb
        if extended_count <= 2 and all(d < 0.08 for d in thumb_distances.values()):
            return ("O", 0.85)
        
        # C - Curved hand forming C shape (fingers curve toward thumb but don't touch)
        if extended_count >= 3:
            avg_dist = sum(thumb_distances.values()) / 4
            if 0.08 < avg_dist < 0.18:
                return ("C", 0.78)
        
        # ========== FIST VARIATIONS (A, S, E, M, N, T, X) ==========
        
        # X - Index bent in hook shape
        if is_index_hooked() and not finger_states["middle"] and not finger_states["ring"] and not finger_states["pinky"]:
            return ("X", 0.80)
        
        # Check if it's a fist (most fingers folded)
        if extended_count <= 1:
            # T - Thumb between index and middle (thumb tip near index PIP)
            if dist(thumb_tip, index_pip) < 0.06:
                return ("T", 0.80)
            
            # E - Fingertips curl to touch thumb
            if all(d < 0.09 for d in list(thumb_distances.values())):
                return ("E", 0.78)
            
            # M - Three fingers (index, middle, ring) over thumb
            # Check if thumb is hidden under fingers
            if thumb_tip[1] > index_pip[1] and thumb_tip[1] > middle_pip[1]:
                # Approximate M by checking thumb is under multiple fingers
                if dist(thumb_tip, middle_mcp) < 0.1:
                    return ("M", 0.75)
            
            # N - Two fingers (index, middle) over thumb  
            if thumb_tip[1] > index_pip[1] and thumb_tip[0] > index_mcp[0] - 0.05:
                if dist(thumb_tip, index_mcp) < 0.08:
                    return ("N", 0.75)
            
            # S - Fist with thumb in front of fingers
            if thumb_tip[1] < index_pip[1] and thumb_tip[1] < middle_pip[1]:
                # Thumb is in front (above in y) of the fingers
                return ("S", 0.80)
            
            # A - Fist with thumb alongside (default fist)
            if abs(thumb_tip[1] - index_mcp[1]) < 0.1:
                return ("A", 0.82)
        
        # ========== SPECIAL CASES ==========
        
        # J - Pinky extended with specific position (simplified static version)
        if (not finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            finger_states["pinky"] and
            finger_states["thumb"]):
            # J is like I but with thumb out - treat as J
            return ("J", 0.75)
        
        # Z - Index pointing (simplified static version)
        # This is a motion letter, so we approximate with index pointing forward
        if (finger_states["index"] and 
            not finger_states["middle"] and 
            not finger_states["ring"] and 
            not finger_states["pinky"] and
            abs(index_tip[2]) > 0.05):  # Z-depth indicates pointing forward
            return ("Z", 0.70)
        
        return (None, 0.0)
    
    def recognize(self, frame) -> Dict:
        """
        Recognize hand sign in the given frame.
        
        Args:
            frame: BGR image from camera (numpy array)
            
        Returns:
            Dict with keys:
                - character: Recognized character or None
                - confidence: Recognition confidence (0.0-1.0)
                - hand_detected: Whether a hand was detected
                - landmarks: Hand landmarks if detected
                - handedness: "Left" or "Right" or None
        """
        result = {
            "character": None,
            "confidence": 0.0,
            "hand_detected": False,
            "landmarks": None,
            "handedness": None
        }
        
        if frame is None:
            return result
            
        if not self.mediapipe_available:
            return result
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = self.hands.process(rgb_frame)
        
        if not results.multi_hand_landmarks:
            # Reset stability tracking when no hand detected
            self.last_gesture = None
            self.gesture_count = 0
            return result
        
        # Get first hand
        hand_landmarks = results.multi_hand_landmarks[0]
        handedness = results.multi_handedness[0].classification[0].label
        
        result["hand_detected"] = True
        result["landmarks"] = hand_landmarks
        result["handedness"] = handedness
        
        # Classify the gesture
        gesture, confidence = self._classify_gesture(hand_landmarks, handedness)
        
        if gesture is None:
            self.last_gesture = None
            self.gesture_count = 0
            return result
        
        # Stability check - require consistent detection across frames
        if gesture == self.last_gesture:
            self.gesture_count += 1
        else:
            self.last_gesture = gesture
            self.gesture_count = 1
        
        # Only confirm gesture after stability threshold
        if self.gesture_count >= config.STABILITY_FRAMES:
            # Map gesture to output character
            output_char = None
            
            # Check alphabets
            if gesture in self.mappings.get("alphabets", {}):
                output_char = self.mappings["alphabets"][gesture]
            # Check numbers
            elif gesture in self.mappings.get("numbers", {}):
                output_char = self.mappings["numbers"][gesture]
            # Check nepali
            elif gesture in self.mappings.get("nepali", {}):
                output_char = self.mappings["nepali"][gesture]
            else:
                output_char = gesture  # Use raw gesture name if no mapping
            
            result["character"] = output_char
            result["confidence"] = confidence
            
            if config.DEBUG_MODE:
                print(f"[DEBUG] Recognized: {gesture} -> {output_char} (conf: {confidence:.2f})")
        
        return result
    
    def draw_landmarks(self, frame, landmarks) -> np.ndarray:
        """
        Draw hand landmarks on the frame.
        
        Args:
            frame: BGR image
            landmarks: MediaPipe hand landmarks
            
        Returns:
            Frame with landmarks drawn
        """
        if landmarks is None or not self.mediapipe_available:
            return frame
        
        annotated = frame.copy()
        self.mp_drawing.draw_landmarks(
            annotated,
            landmarks,
            self.mp_hands.HAND_CONNECTIONS,
            self.mp_drawing_styles.get_default_hand_landmarks_style(),
            self.mp_drawing_styles.get_default_hand_connections_style()
        )
        return annotated
    
    def close(self):
        """Release MediaPipe resources."""
        self.hands.close()
        print("Recognizer closed")
    
    def __del__(self):
        """Destructor to ensure resources are released."""
        try:
            self.hands.close()
        except:
            pass
