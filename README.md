# ✋ Sunana — Real-Time Hand Sign Recognition System

A real-time hand sign recognition engine that detects ASL (American Sign Language) alphabets and numbers using Python and MediaPipe.

> Built with a focus on real-time performance, modular architecture, and extensibility.

---

## 📸 Demo

> Real-time hand sign detection using webcam

<p align="center">
  <img src="assets/2026-04-2923-56-46-ezgif.com-video-to-gif-converter.gif" width="600"/>
</p>

---

## 🚀 Features

- 🔴 Real-time gesture recognition via webcam  
- 🔤 Supports ASL alphabets (A–Z)  
- 🔢 Supports numbers (0–9)  
- 🧠 Custom gesture classification engine  
- ⚙️ Multiple modes:
  - GUI preview
  - Console-only mode
  - REST API server  
- 🧩 Modular structure for easy extension  

---

## 🧠 How It Works

```
Camera Input
   ↓
MediaPipe Hand Tracking
   ↓
Landmark Extraction
   ↓
Gesture Classification Engine
   ↓
Output (GUI / Console / API)
```

The system uses MediaPipe to detect hand landmarks and processes them through a custom-built classification engine to recognize gestures in real time.

---

## 🛠️ Installation

1. Install Python **3.8 – 3.11**  
   *(MediaPipe does not support Python 3.12+ yet)*

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## ▶️ Usage

### Run with GUI (Default)

```bash
python app.py
```

Shows webcam feed with real-time recognition. Press `Q` to quit.

---

### Run without GUI

```bash
python app.py --no-gui
```

Outputs recognized gestures in the console.

---

### Run API Server

```bash
python app.py --api
```

Starts a server on `http://localhost:5000`

---

## 🌐 API Endpoints

- `GET /health` → Check server status  
- `POST /recognize` → Recognize gesture from base64 image  

---

## ⚙️ Configuration

Edit `config.py`:

- `CAMERA_INDEX` → Camera selection  
- `MIN_DETECTION_CONFIDENCE` → Detection sensitivity  
- `STABILITY_FRAMES` → Frames required for stable detection  
- `OUTPUT_DELAY_MS` → Delay between outputs  

---

## 🧩 Project Structure

```
.
├── app.py
├── camera.py
├── gesture_recognizer.py
├── config.py
├── requirements.txt
├── data/
│   └── gesture_mappings.json
├── assets/
│   └── demo.gif
└── README.md
```

---

## 📈 Future Improvements

- 🌍 Support for Nepali Sign Language  
- 🌐 Web interface using API  
- 📱 Mobile integration  
- 🤖 Machine learning-based classification  

---

## 💡 Use Cases

- Accessibility tools for hearing-impaired users  
- Gesture-based human-computer interaction  
- Educational tools for learning sign language  

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the project and improve it.

---

## 📄 License

MIT License
