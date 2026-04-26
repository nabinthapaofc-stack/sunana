"""
Text-to-Speech Engine for Hand Sign Recognition
Uses Windows SAPI via PowerShell for reliable TTS
"""

import subprocess
import threading
import time


class TextToSpeech:
    """
    Text-to-speech engine using Windows SAPI via PowerShell.
    More reliable when running alongside OpenCV.
    """
    
    def __init__(self):
        """Initialize the TTS engine."""
        try:
            import config
            self.enabled = getattr(config, 'ENABLE_TTS', True)
            self.rate = getattr(config, 'TTS_RATE', 150)
        except:
            self.enabled = True
            self.rate = 150
        
        self.available = True
        self._last_speak_time = 0
        self._min_interval = 0.3  # Minimum seconds between speaks
        
        if not self.enabled:
            print("! TTS disabled in config")
            return
        
        print("✓ Text-to-Speech engine initialized (Windows SAPI)")
    
    def _speak_async(self, text: str):
        """Speak text using PowerShell in background."""
        try:
            # Use PowerShell to call Windows SAPI
            sapi_rate = max(-10, min(10, (self.rate - 150) // 25))
            
            ps_command = f'''
            Add-Type -AssemblyName System.Speech
            $synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
            $synth.Rate = {sapi_rate}
            $synth.Speak("{text}")
            '''
            
            subprocess.run(
                ["powershell", "-Command", ps_command],
                capture_output=True,
                timeout=5
            )
        except Exception as e:
            pass  # Silently ignore errors
    
    def speak(self, text: str):
        """
        Speak text without blocking the main loop.
        
        Args:
            text: The text to speak (e.g., a letter or number)
        """
        if not self.available or not self.enabled:
            return
        
        # Rate limit to avoid overlapping
        current_time = time.time()
        if current_time - self._last_speak_time < self._min_interval:
            return
        
        self._last_speak_time = current_time
        print(f"[TTS] Speaking: {text}")
        
        # Run in background thread - each speak creates new thread
        t = threading.Thread(target=self._speak_async, args=(text,), daemon=True)
        t.start()
    
    def stop(self):
        """Stop the TTS engine."""
        print("TTS engine stopped")
    
    def __del__(self):
        """Destructor."""
        pass
