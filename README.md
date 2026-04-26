# Hand Sign Recognition System

Real-time ASL (American Sign Language) alphabet and number recognition using Python and MediaPipe.

## Features

- **Real-time Recognition**: Detects hand signs from webcam in real-time
- **ASL Alphabets (A-Z)**: Recognizes standard ASL finger spelling
- **Numbers (0-9)**: Recognizes standard hand number signs
- **Multiple Modes**: GUI preview, console-only, or API server
- **Extensible**: Easy to add more characters (e.g., Nepali alphabets)

## Installation

1. **Important**: Install Python **3.8 to 3.11** (MediaPipe does not yet support 3.12+)
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

### With Camera Preview (Default)
```bash
python app.py
```
Shows a window with camera feed and recognized signs. Press 'Q' to quit.

### Console Only Mode
```bash
python app.py --no-gui
```
Prints recognized characters to console without visual preview. Press Ctrl+C to quit.

### API Server Mode
```bash
python app.py --api
```
Starts a Flask server on port 5000 with endpoints:
- `GET /health` - Health check
- `POST /recognize` - Recognize sign from base64 image

## Project Structure

```
├── app.py                    # Main application entry point
├── camera.py                 # Camera capture module
├── gesture_recognizer.py     # Core ASL recognition engine
├── config.py                 # Configuration settings
├── requirements.txt          # Python dependencies
├── README.md                 # This file
└── data/
    └── gesture_mappings.json # Character output mappings
```

## Supported Signs

### Numbers
| Sign | Description |
|------|-------------|
| 0 | O shape - fingers touch thumb |
| 1 | Only index finger extended |
| 2 | V sign - index and middle spread |
| 3 | Thumb, index, middle extended |
| 4 | Four fingers up, thumb folded |
| 5 | Open palm - all fingers spread |

### Letters (Selection)
| Sign | Description |
|------|-------------|
| A | Fist with thumb alongside |
| B | Four fingers up straight, thumb across |
| L | Index and thumb form L shape |
| V | Index and middle spread (peace sign) |
| Y | Thumb and pinky extended |

## Configuration

Edit `config.py` to adjust settings:
- `CAMERA_INDEX`: Which camera to use (0 = default)
- `MIN_DETECTION_CONFIDENCE`: Detection sensitivity (0.0-1.0)
- `STABILITY_FRAMES`: Frames needed to confirm gesture
- `OUTPUT_DELAY_MS`: Minimum delay between outputs

## Extending

To add new characters:
1. Add recognition logic in `gesture_recognizer.py` under `_classify_gesture()`
2. Add mapping in `data/gesture_mappings.json`

## License

MIT License
