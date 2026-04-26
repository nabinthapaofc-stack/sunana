
import sys
print(f"Python executing: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import mediapipe as mp
    print(f"MediaPipe file: {mp.__file__}")
    print(f"Dir(mp): {dir(mp)}")
    
    if hasattr(mp, 'solutions'):
        print("mp.solutions exists")
        print(f"mp.solutions.hands: {mp.solutions.hands}")
    else:
        print("ERROR: mp.solutions does NOT exist")
        # specific check for bad install
        import importlib
        try:
             importlib.import_module("mediapipe.python.solutions")
             print("But mediapipe.python.solutions IS importable directly")
        except Exception as e:
             print(f"And mediapipe.python.solutions is NOT importable: {e}")

except Exception as e:
    print(f"CRITICAL ERROR importing mediapipe: {e}")
