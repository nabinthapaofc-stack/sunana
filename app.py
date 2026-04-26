"""
Hand Sign Recognition System - Main Application
Real-time ASL alphabet and number recognition from webcam

Usage:
    python app.py           # Run with camera preview
    python app.py --no-gui  # Run without camera preview (console only)
    python app.py --api     # Run as Flask API server
"""

import cv2
import time
import argparse
from camera import Camera
from gesture_recognizer import HandSignRecognizer
from tts_engine import TextToSpeech
import config


def run_recognition_loop():
    """
    Main recognition loop with camera preview.
    Press 'q' to quit.
    """
    print("\n" + "="*50)
    print("  HAND SIGN RECOGNITION SYSTEM")
    print("  Press 'Q' to quit")
    print("="*50 + "\n")
    
    recognizer = HandSignRecognizer()
    camera = Camera()
    tts = TextToSpeech()
    
    # Track last output to avoid spamming
    last_output_char = None
    last_output_time = 0
    
    # FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    try:
        if not camera.open():
            print("Failed to open camera. Exiting.")
            return
        
        print("Starting recognition... Show your hand signs!\n")
        
        while True:
            # Capture frame
            frame = camera.capture_frame()
            if frame is None:
                print("Failed to capture frame")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Recognize gesture
            result = recognizer.recognize(frame)
            
            # Draw landmarks if hand detected
            if result["hand_detected"] and config.SHOW_LANDMARKS:
                frame = recognizer.draw_landmarks(frame, result["landmarks"])
            
            # Output recognized character
            if result["character"] is not None:
                current_time = time.time() * 1000  # Convert to ms
                
                # Only output if different character or enough time passed
                if (result["character"] != last_output_char or 
                    current_time - last_output_time > config.OUTPUT_DELAY_MS):
                    
                    if config.CONSOLE_OUTPUT:
                        print(f"✓ Recognized: {result['character']} (confidence: {result['confidence']:.2f})")
                    
                    # Speak the recognized character
                    tts.speak(result['character'])
                    
                    last_output_char = result["character"]
                    last_output_time = current_time
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count >= 10:
                elapsed = time.time() - fps_start_time
                current_fps = fps_frame_count / elapsed
                fps_start_time = time.time()
                fps_frame_count = 0
            
            # Add text overlay
            if config.SHOW_CAMERA_PREVIEW:
                # Display current recognized character
                if result["character"]:
                    cv2.putText(
                        frame, 
                        f"Sign: {result['character']}", 
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.2, 
                        (0, 255, 0), 
                        2
                    )
                else:
                    status = "Hand detected" if result["hand_detected"] else "Show your hand"
                    cv2.putText(
                        frame, 
                        status, 
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, 
                        (200, 200, 200), 
                        2
                    )
                
                # Display FPS
                if config.SHOW_FPS:
                    cv2.putText(
                        frame, 
                        f"FPS: {current_fps:.1f}", 
                        (frame.shape[1] - 120, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.6, 
                        (255, 255, 0), 
                        2
                    )
                
                # Show the frame
                cv2.imshow(config.WINDOW_NAME, frame)
            
            # Check for quit key
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                print("\nQuitting...")
                break
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        camera.close()
        recognizer.close()
        tts.stop()
        cv2.destroyAllWindows()
        print("Cleanup complete")


def run_console_only():
    """
    Run recognition without GUI - console output only.
    Press Ctrl+C to quit.
    """
    print("\n" + "="*50)
    print("  HAND SIGN RECOGNITION (Console Mode)")
    print("  Press Ctrl+C to quit")
    print("="*50 + "\n")
    
    recognizer = HandSignRecognizer()
    camera = Camera()
    
    last_output_char = None
    last_output_time = 0
    
    try:
        if not camera.open():
            print("Failed to open camera. Exiting.")
            return
        
        print("Starting recognition...\n")
        
        while True:
            frame = camera.capture_frame()
            if frame is None:
                continue
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            result = recognizer.recognize(frame)
            
            if result["character"] is not None:
                current_time = time.time() * 1000
                
                if (result["character"] != last_output_char or 
                    current_time - last_output_time > config.OUTPUT_DELAY_MS):
                    
                    print(f"Recognized: {result['character']}")
                    last_output_char = result["character"]
                    last_output_time = current_time
            
            # Small delay to prevent CPU overload
            time.sleep(0.01)
    
    except KeyboardInterrupt:
        print("\nQuitting...")
    
    finally:
        camera.close()
        recognizer.close()


def run_api_server():
    """
    Run as Flask API server for integration.
    """
    try:
        from flask import Flask, request, jsonify
        from flask_cors import CORS
        import base64
        import numpy as np
    except ImportError:
        print("Flask not installed. Run: pip install flask flask-cors")
        return
    
    app = Flask(__name__)
    CORS(app)
    
    recognizer = HandSignRecognizer()
    
    @app.route('/health', methods=['GET'])
    def health():
        """Health check endpoint."""
        return jsonify({
            "status": "healthy",
            "service": "hand-sign-recognition"
        })
    
    @app.route('/recognize', methods=['POST'])
    def recognize():
        """
        Recognize hand sign from base64 image.
        
        Request body:
            {"image": "base64_encoded_image_data"}
        
        Response:
            {
                "success": true,
                "character": "A",
                "confidence": 0.95,
                "hand_detected": true
            }
        """
        try:
            data = request.get_json()
            
            if 'image' not in data:
                return jsonify({"success": False, "error": "No image provided"}), 400
            
            # Decode base64 image
            image_data = base64.b64decode(data['image'])
            nparr = np.frombuffer(image_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                return jsonify({"success": False, "error": "Invalid image data"}), 400
            
            # Recognize
            result = recognizer.recognize(frame)
            
            return jsonify({
                "success": True,
                "character": result["character"],
                "confidence": result["confidence"],
                "hand_detected": result["hand_detected"],
                "handedness": result["handedness"]
            })
        
        except Exception as e:
            return jsonify({"success": False, "error": str(e)}), 500
    
    print("\n" + "="*50)
    print("  HAND SIGN RECOGNITION API SERVER")
    print("  Endpoints:")
    print("    GET  /health   - Health check")
    print("    POST /recognize - Recognize from image")
    print("="*50 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=False)


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Hand Sign Recognition System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python app.py           # Run with camera preview (default)
    python app.py --no-gui  # Run without preview, console only
    python app.py --api     # Run as Flask API server
        """
    )
    
    parser.add_argument(
        '--no-gui', 
        action='store_true',
        help='Run without camera preview (console output only)'
    )
    
    parser.add_argument(
        '--api', 
        action='store_true',
        help='Run as Flask API server'
    )
    
    args = parser.parse_args()
    
    if args.api:
        run_api_server()
    elif args.no_gui:
        run_console_only()
    else:
        run_recognition_loop()


if __name__ == "__main__":
    main()
