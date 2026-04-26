"""Quick test script for TTS"""
import pyttsx3

print("Testing pyttsx3...")

try:
    engine = pyttsx3.init()
    print("Engine initialized")
    
    engine.setProperty('rate', 150)
    engine.setProperty('volume', 1.0)
    
    print("Speaking 'Hello'...")
    engine.say("Hello")
    engine.runAndWait()
    
    print("Speaking 'A'...")
    engine.say("A")
    engine.runAndWait()
    
    print("Done!")
    engine.stop()
    
except Exception as e:
    print(f"Error: {e}")
