# LightHive
## AI-Powered Intelligent Stage Lighting and Music Visualization

LightHive is an AI-powered stage lighting visualization system that analyzes music, predicts lighting moods, detects beats, and uses camera-based gestures and pose tracking to create an interactive lighting experience.

The project combines a React frontend with a Python backend to integrate audio analysis, computer vision, and dynamic lighting visualization.

## Features

### Audio Analysis
- Upload and analyze music files.
- Extract audio features for music-aware lighting.
- Detect beats and synchronize visual effects with music playback.
- Analyze musical characteristics such as energy, valence, and arousal.

### AI-Based Lighting Prediction
- Predict lighting colors based on analyzed audio.
- Generate RGB color values for lighting visualization.
- Map predicted moods to lighting palettes.
- Support Calm, Dreamy, Energetic, and Intense moods.
- Adjust lighting intensity according to music features.

### Beat-Synchronized Visualization
- Synchronize lighting effects with music playback.
- Animate the stage visualization using beat pulses.
- Dynamically update lighting colors as a song progresses.

### Gesture-Controlled Lighting
- Detect gestures using a camera feed.
- Pause automatic lighting through a supported gesture.
- Trigger climax and full-stage reveal effects.
- Control directional spotlight effects using recognized gestures.

### Performer Pose Detection
- Detect performer poses using a pose-estimation model.
- Process pose landmarks for interactive stage experiences.

### Audience Analysis
- Analyze camera frames for audience-related information.
- Display available audience metrics.
- Use audience-analysis results to influence lighting intensity where supported.

### Interactive Web Interface
- Upload and play music.
- View predicted lighting moods, colors, and intensity.
- Control camera-based features.
- Visualize stage lighting effects in real time.

## Technology Stack

| Component | Technology |
|---|---|
| Frontend | React, JavaScript, CSS |
| Backend | Python, FastAPI |
| Audio analysis | Python audio-processing pipeline |
| Beat detection | Backend beat-detection endpoint |
| Computer vision | YOLO pose model and camera-frame analysis |
| API communication | HTTP, JSON, multipart form data |
| Visualization | React-based stage and lighting effects |

## Project Structure

```text
LightHive/
├── backend.py
├── feature2/
│   └── predict.py
├── website/
│   └── src/
│       └── App.jsx
├── data/
├── yolo11n-pose.pt
├── audio_pipeline_result.json
└── README.md
```

This is a simplified overview of the project structure. Other files and directories may be present in the repository.

## Prerequisites

Before running LightHive, ensure you have:

- Python installed with the required backend dependencies.
- Node.js and npm.
- Git.
- A modern web browser.
- A compatible camera for gesture, pose, and audience analysis.
- Required model weights and data assets.

## Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd LightHive
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Set up the Python environment

Create a virtual environment from the project root.

**Windows**

```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS and Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the backend dependencies if a `requirements.txt` file is available:

```bash
pip install -r requirements.txt
```

If the dependency file does not exist, install the packages required by the backend and its audio-processing and computer-vision modules.

### 3. Start the backend

From the project root, run:

```bash
uvicorn backend:app --reload --host 127.0.0.1 --port 8000
```

This command assumes that `backend.py` exposes a FastAPI application named `app`. Adjust it if your backend uses a different entry point.

The backend should be available at:

```text
http://127.0.0.1:8000
```

If FastAPI documentation is enabled, access it at:

```text
http://127.0.0.1:8000/docs
```

### 4. Set up the frontend

Open a second terminal and navigate to the frontend directory:

```bash
cd website
npm install
```

Start the development server:

```bash
npm run dev
```

If the project uses a different start script, check `website/package.json`.

Open the local URL displayed in the terminal.

## Backend API Integration

The frontend integrates with backend endpoints for audio analysis, lighting prediction, and computer vision.

| Endpoint | Purpose |
|---|---|
| `GET /state` | Retrieve the current lighting state |
| `POST /analyze` | Analyze an uploaded audio file |
| `POST /detect-beats` | Detect beats in the audio |
| `POST /predict-lighting` | Predict lighting characteristics |
| `POST /detect-gesture` | Detect gestures from camera frames |
| `POST /detect-pose` | Detect performer pose landmarks |
| `POST /analyze-audience` | Analyze audience-related information |

These endpoints reflect the frontend integrations. Refer to the backend implementation for the exact request formats, response fields, and supported behavior.

## How to Use LightHive

1. Start the Python backend.
2. Start the React frontend.
3. Open the application in your browser.
4. Upload a music track.
5. Allow the audio-analysis pipeline to process the track.
6. Play the track and observe the synchronized lighting visualization.
7. Explore the available lighting moods and color palettes.
8. Enable the camera and grant permission when prompted.
9. Test the supported gestures and performer pose tracking.
10. Explore audience-analysis metrics where available.

Camera-based features require a compatible camera and appropriate browser permissions.

## Performance Considerations

- Audio-analysis performance depends on the input file and processing pipeline.
- Camera-based detection performance depends on the hardware, camera quality, and model complexity.
- Real-time responsiveness depends on backend processing latency and frontend update frequency.
- Running multiple computer-vision processes simultaneously may increase CPU or GPU usage.

## Security and Privacy

- Do not commit API keys, passwords, or other credentials.
- Review generated analysis files before publishing them.
- Avoid committing private recordings or sensitive datasets.
- Enable camera access only when needed.
- Review audio and camera data handling before using the application with performers or audiences.
- Do not expose a development server publicly without appropriate security controls.

## Current Limitations

- Lighting predictions depend on the quality of the audio features and prediction model.
- Gesture and pose detection can be affected by camera positioning, occlusion, and lighting conditions.
- Some features may require additional model weights or datasets.
- The visualization does not automatically control physical lighting fixtures unless suitable hardware integration is implemented.

## Future Enhancements

- Integration with DMX or Art-Net lighting systems.
- Improved beat tracking and music-section recognition.
- Configurable gesture mappings.
- More robust performer tracking.
- Adaptive lighting based on audience engagement.
- Optimized inference and reduced processing latency.
- Saving, exporting, and replaying lighting scenes.
- Production deployment and improved configuration management.

## Contributing

Contributions and suggestions are welcome.

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Test the frontend and backend.
5. Submit a pull request describing your changes.

## License

No license has been specified yet. Add a `LICENSE` file before distributing the project under an open-source license.

## Project Summary

LightHive combines music analysis, AI-based lighting prediction, beat synchronization, gesture recognition, performer pose detection, and audience analysis to explore more interactive stage-lighting experiences.
