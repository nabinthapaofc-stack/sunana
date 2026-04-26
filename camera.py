"""
Camera module - Handles webcam capture and initialization
Provides simple interface to capture frames from laptop webcam
"""

import cv2
import time
import config


class Camera:
    """
    Manages laptop webcam access and frame capture.
    
    Usage:
        # Method 1: Context manager (recommended)
        with Camera() as camera:
            frame = camera.capture_frame()
        
        # Method 2: Manual control
        camera = Camera()
        camera.open()
        frame = camera.capture_frame()
        camera.close()
    """
    
    def __init__(self):
        """Initialize camera object (doesn't open camera yet)."""
        self.camera = None
        self.is_open = False
    
    def open(self):
        """
        Open the webcam connection.
        
        Returns:
            bool: True if successful, False otherwise
        """
        if self.is_open:
            return True
        
        try:
            # Open the camera at specified index
            self.camera = cv2.VideoCapture(config.CAMERA_INDEX)
            
            if not self.camera.isOpened():
                print(f"Error: Could not open camera at index {config.CAMERA_INDEX}")
                return False
            
            # Set camera resolution
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, config.FRAME_WIDTH)
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, config.FRAME_HEIGHT)
            
            # Warm up the camera
            print(f"Warming up camera ({config.CAMERA_WARMUP_TIME}s)...")
            time.sleep(config.CAMERA_WARMUP_TIME)
            
            # Discard initial frames (often overexposed)
            for _ in range(5):
                self.camera.read()
            
            self.is_open = True
            print("✓ Camera ready")
            return True
            
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return False
    
    def capture_frame(self):
        """
        Capture a single frame from the webcam.
        
        Returns:
            numpy.ndarray: BGR frame if successful, None if failed
        """
        if not self.is_open:
            print("Error: Camera not open. Call open() first.")
            return None
        
        try:
            ret, frame = self.camera.read()
            
            if not ret or frame is None:
                print("Error: Failed to capture frame")
                return None
            
            return frame
            
        except Exception as e:
            print(f"Error capturing frame: {e}")
            return None
    
    def capture_frame_rgb(self):
        """
        Capture a single frame and convert to RGB format.
        Used for MediaPipe which expects RGB input.
        
        Returns:
            numpy.ndarray: RGB frame if successful, None if failed
        """
        frame = self.capture_frame()
        if frame is not None:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return None
    
    def close(self):
        """Close the webcam connection and release resources."""
        if self.camera is not None:
            self.camera.release()
            self.camera = None
        self.is_open = False
        print("Camera closed")
    
    def __enter__(self):
        """Context manager entry - opens camera."""
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - closes camera automatically."""
        self.close()


def capture_single_frame():
    """
    Convenience function to capture one frame and close camera.
    
    Returns:
        numpy.ndarray: BGR frame or None if failed
    """
    with Camera() as camera:
        return camera.capture_frame()
