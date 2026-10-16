
import { useEffect, useRef, useState } from "react";
import "./App.css";

export default function App() {
  const [playing, setPlaying] = useState(false);
  const [beatData, setBeatData] = useState(null);
  const audioRef = useRef(null);
const lastBeatIndexRef = useRef(-1);
const [beatPulse, setBeatPulse] = useState(false);
  const [intensity, setIntensity] = useState(70);
  const [mood, setMood] = useState("ENERGETIC");
 const [track, setTrack] = useState(null);
const [audioUrl, setAudioUrl] = useState("");
const [color, setColor] = useState("#8b5cf6");
  const [cameraMode, setCameraMode] = useState("performer");
const [cameraStatus, setCameraStatus] = useState("Camera off");
const [cameraError, setCameraError] = useState("");
const videoRef = useRef(null);
const cameraStreamRef = useRef(null);
const [detectedGesture, setDetectedGesture] = useState(null);
const [gestureAction, setGestureAction] = useState("Waiting for gesture");
const [spotlightDirection, setSpotlightDirection] = useState(null);
const [autoLightingPaused, setAutoLightingPaused] = useState(false);
const [climaxActive, setClimaxActive] = useState(false);
const [fullRevealActive, setFullRevealActive] = useState(false);
const [gestureStatus, setGestureStatus] = useState("Waiting for camera");
const [gestureEnabled, setGestureEnabled] = useState(false);
const scanInProgress = useRef(false);
const [poseLandmarks, setPoseLandmarks] = useState(null);
const smoothedPoseRef = useRef(null);
const [poseStatus, setPoseStatus] = useState("Pose tracking off");
const poseScanInProgress = useRef(false);
  
const [apiState, setApiState] = useState(null);
const [apiStatus, setApiStatus] = useState("Checking");
const [audienceData, setAudienceData] = useState(null);
const [audienceStatus, setAudienceStatus] = useState("Waiting for camera");
const audienceScanInProgress = useRef(false);


async function fetchLightingState() {
  try {
    const response = await fetch("http://127.0.0.1:8000/state");
    if (!response.ok) throw new Error("API request failed");

    const data = await response.json();
    setApiState(data);
    setApiStatus("Connected");

    if (data.rgb) {
      setColor(`rgb(${data.rgb.r}, ${data.rgb.g}, ${data.rgb.b})`);
    }

    setIntensity(Math.round((data.brightness ?? 0.6) * 100));
  } catch (error) {
    setApiStatus("Disconnected");
    console.error("Python API error:", error);
  }
}

  
const [audioAnalysis, setAudioAnalysis] = useState(null);
const [audioStatus, setAudioStatus] = useState("No song analyzed");
useEffect(() => {
  if (!track) {
    setAudioUrl("");
    return;
  }

  const url = URL.createObjectURL(track);
  setAudioUrl(url);

  return () => {
    URL.revokeObjectURL(url);
  };
}, [track]);



const handleUpload = async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

 audioRef.current?.pause();
 lastBeatIndexRef.current = -1;
setPlaying(false);
setBeatPulse(false);
setTrack(file);
setBeatData(null);
setAudioAnalysis(null);
setAudioStatus("Analysing song...");

  const formData = new FormData();
  formData.append("file", file);

  // Run beat detection separately from lighting prediction
  const beatPromise = fetch(
    "http://127.0.0.1:8000/detect-beats",
    {
      method: "POST",
      body: formData,
    }
  )
    .then(async (response) => {
      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.detail || "Beat detection failed");
      }

      setBeatData(result);
      console.log("Beat detection result:", result);
    })
    .catch((error) => {
      console.error("Beat detection error:", error);
    });

  try {
    const response = await fetch(
      "http://127.0.0.1:8000/predict-lighting",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Prediction failed");
    }

    const rgb = result.lighting.rgb;
    const predictedColor = `rgb(${rgb.r}, ${rgb.g}, ${rgb.b})`;

    setColor(predictedColor);
    setApiState(result.lighting);
    setIntensity(
      Math.round((result.lighting.brightness ?? 0.6) * 100)
    );

    const predictedMood =
      result.valence > 0.35
        ? "ENERGETIC"
        : result.valence < -0.35
          ? "DREAMY"
          : "CALM";

    setMood(predictedMood);
    setAudioAnalysis(result);
    setAudioStatus("AI lighting applied successfully!");

  } catch (error) {
    console.error("Lighting prediction error:", error);
    setAudioStatus(error.message || "Could not analyse song");
  }

  await beatPromise;
};

async function startCamera() {
  setCameraError("");

  if (!navigator.mediaDevices?.getUserMedia) {
    setCameraError("Camera unavailable. Use localhost or HTTPS.");
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: true,
      audio: false,
    });

    cameraStreamRef.current = stream;

    if (videoRef.current) {
      videoRef.current.srcObject = stream;
      await videoRef.current.play();
    }

    setCameraStatus("Camera connected");
  } catch (error) {
    setCameraStatus("Camera off");
    setCameraError(
      error.name === "NotAllowedError"
        ? "Camera permission denied. Allow camera access in your browser."
        : "Could not start the camera. Check that it is connected."
    );
  }
}

function stopCamera() {
  cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
  cameraStreamRef.current = null;

  if (videoRef.current) {
    videoRef.current.srcObject = null;
  }

  setCameraStatus("Camera off");
}

useEffect(() => {
  return () => {
    cameraStreamRef.current?.getTracks().forEach((track) => track.stop());
  };
}, []);
async function analyzeGestureFrame() {
  const video = videoRef.current;

  if (!video || video.readyState < 2 || scanInProgress.current) {
    return;
  }

  scanInProgress.current = true;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7)
    );

    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "webcam.jpg");

    const response = await fetch(
      "http://127.0.0.1:8000/detect-gesture",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Gesture detection failed");
    }
    




if (result.detected) {
  const gesture = result.detected;
  const eventName =
    typeof gesture === "string" ? gesture : gesture.event;

  setDetectedGesture(gesture);
  setGestureStatus("Gesture detected");

  if (eventName === "PAUSE_AUTO") {
    setAutoLightingPaused(true);
    setGestureAction("Automatic lighting paused");
    setClimaxActive(false);
    setFullRevealActive(false);
  } else if (eventName === "CLIMAX") {
    setAutoLightingPaused(false);
    setClimaxActive(true);
    setFullRevealActive(false);
    setColor("#ffffff");
    setIntensity(100);
    setGestureAction("CLIMAX: Dramatic stage flash");
  } else if (eventName === "FULL_STAGE_REVEAL") {
    setAutoLightingPaused(false);
    setClimaxActive(false);
    setFullRevealActive(true);
    setIntensity(100);
    setGestureAction("FULL STAGE REVEAL: All beams active");
  } else if (eventName === "SPOTLIGHT_DIRECTION") {
    setSpotlightDirection(gesture.direction || null);
    setGestureAction("SPOTLIGHT: Direction updated");
  } else {
    setGestureAction(`Gesture detected: ${eventName || "Unknown"}`);
  }
} else {
  setGestureStatus("Scanning for gestures");
}


  } catch (error) {
    console.error("Gesture detection error:", error);
    setGestureStatus(error.message || "Analysis error");
  } finally {
    scanInProgress.current = false;
  }
}

async function analyzeAudienceFrame() {
  const video = videoRef.current;

  if (
    cameraMode !== "audience" ||
    !video ||
    video.readyState < 2 ||
    audienceScanInProgress.current
  ) {
    return;
  }

  audienceScanInProgress.current = true;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    const context = canvas.getContext("2d");
    if (!context) return;

    context.drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.6)
    );

    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "audience.jpg");

    const response = await fetch(
      "http://127.0.0.1:8000/analyze-audience",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Audience analysis failed");
    }

    setAudienceData(result);
    setAudienceStatus("Audience analysis active");
  } catch (error) {
    console.error("Audience analysis error:", error);
    setAudienceStatus(error.message || "Audience analysis failed");
  } finally {
    audienceScanInProgress.current = false;
  }
}

async function analyzePoseFrame() {
  const video = videoRef.current;

  if (
    !video ||
    video.readyState < 2 ||
    poseScanInProgress.current
  ) {
    return;
  }

  poseScanInProgress.current = true;

  try {
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    canvas.getContext("2d").drawImage(video, 0, 0);

    const blob = await new Promise((resolve) =>
      canvas.toBlob(resolve, "image/jpeg", 0.7)
    );

    if (!blob) return;

    const formData = new FormData();
    formData.append("file", blob, "pose-frame.jpg");

    const response = await fetch(
      "http://127.0.0.1:8000/detect-pose",
      {
        method: "POST",
        body: formData,
      }
    );

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.detail || "Pose detection failed");
    }

    if (result.detected && result.landmarks) {
      

const previous = smoothedPoseRef.current;
const alpha = 0.3;

const smoothed = Object.fromEntries(
  Object.entries(result.landmarks).map(([name, point]) => {
    const old = previous?.[name];

    return [
      name,
      {
        x: old ? old.x + alpha * (point.x - old.x) : point.x,
        y: old ? old.y + alpha * (point.y - old.y) : point.y,
        visibility: point.visibility,
      },
    ];
  })
);

smoothedPoseRef.current = smoothed;
setPoseLandmarks(smoothed);
      setPoseStatus("Body detected");
    } else {
      setPoseStatus("Looking for full body...");
    }
  } catch (error) {
    console.error("Pose tracking error:", error);
    setPoseStatus(error.message || "Pose tracking error");
  } finally {
    poseScanInProgress.current = false;
  }
}
useEffect(() => {
  if (cameraStatus !== "Camera connected") return;

  let cancelled = false;
  let timer;

  const scan = async () => {
    if (cancelled) return;

    await analyzePoseFrame();

    if (!cancelled) {
      timer = setTimeout(scan, 150);
    }
  };

  scan();

  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}, [cameraStatus]);
useEffect(() => {
  if (!gestureEnabled || cameraStatus !== "Camera connected") {
    return;
  }

  let cancelled = false;
  let timer;

  const scan = async () => {
    if (cancelled) return;

    await analyzeGestureFrame();

    if (!cancelled) {
      timer = setTimeout(scan, 250);
    }
  };

  scan();

  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}, [gestureEnabled, cameraStatus]);
useEffect(() => {
  if (
    cameraMode !== "audience" ||
    cameraStatus !== "Camera connected"
  ) {
    return;
  }

  let cancelled = false;
  let timer;

  const scan = async () => {
    if (cancelled) return;

    await analyzeAudienceFrame();

    if (!cancelled) {
      timer = setTimeout(scan, 1000);
    }
  };

  scan();

  return () => {
    cancelled = true;
    clearTimeout(timer);
  };
}, [cameraMode, cameraStatus]);
useEffect(() => {
  if (cameraMode !== "audience" || !audienceData) return;
  if (autoLightingPaused || climaxActive) return;

  const energy = Number(audienceData.audience_energy);
  if (!Number.isFinite(energy)) return;

  // Adjust these thresholds if your analyzer uses a different scale.
  if (energy >= 70) {
    setColor("#ff4d8d");
    setIntensity(95);
    setMood("INTENSE");
  } else if (energy >= 40) {
    setColor("#f59e0b");
    setIntensity(75);
    setMood("ENERGETIC");
  } else {
    setColor("#38bdf8");
    setIntensity(40);
    setMood("CALM");
  }
}, [
  audienceData,
  cameraMode,
  autoLightingPaused,
  climaxActive,
]);


async function togglePlayback() {
  const audio = audioRef.current;

  if (!audio || !track) {
    setAudioStatus("Upload a song first.");
    return;
  }

  if (audio.paused) {
    try {
      await audio.play();
    } catch (error) {
      setAudioStatus(
        "Playback was blocked. Press Play in the audio player."
      );
    }
  } else {
    audio.pause();
  }
}
  return (
    <main className="app-shell">
      <header className="topbar">
        <a className="brand" href="#">
          <span className="brand-icon">✳</span>
          LIGHT<span>HIVE</span>
        </a>

        <div className="live-status">
          <span className="status-dot" />
          VIRTUAL STAGE
        </div>

        <span className="header-tag">AI LIGHTING SYSTEM</span>
      </header>

      <section className="dashboard">
        <aside className="sidebar">
          <p className="eyebrow">CONTROL ROOM</p>
          <h1>Feel the music.</h1>
          <p className="intro">
            Intelligent lighting that responds to the energy of sound.
          </p>

          <div className="control-section">
            <label className="section-label">YOUR SOUNDTRACK</label>

            <label className="upload-box">
              <span className="upload-icon">＋</span>
              <strong>{track ? track.name : "Upload a track"}</strong>
              <span>Choose an audio file</span>
              <input
                type="file"
                accept="audio/*"
                onChange={handleUpload}
              />
            </label>

{track && audioUrl && (
  <audio
    ref={audioRef}
    className="audio-player"
    controls
    src={audioUrl}
    onPlay={() => setPlaying(true)}
    onPause={() => {
      setPlaying(false);
      setBeatPulse(false);
    }}
    onEnded={() => {
      setPlaying(false);
      setBeatPulse(false);
    }}
    onTimeUpdate={(event) => {
      const beats = beatData?.beat_times_seconds;

      if (!beats?.length) {
        setBeatPulse(false);
        return;
      }

      const currentTime = event.currentTarget.currentTime;
      let beatIndex = -1;

      for (let i = 0; i < beats.length; i++) {
        if (beats[i] <= currentTime) {
          beatIndex = i;
        } else {
          break;
        }
      }

      if (beatIndex === -1) {
        setBeatPulse(false);
        return;
      }

      const interval =
        beatData?.estimated_beat_interval_seconds || 0.5;

      setBeatPulse(
        currentTime - beats[beatIndex] < interval * 0.3
      );
    }}
  />
)}

{audioStatus && (
  <div className="sidebar-footnote">
    {audioStatus}
  </div>
)}

{audioAnalysis && (
  <div className="control-section">
    <label className="section-label">AI LIGHTING PREDICTION</label>
    <p>Valence: {audioAnalysis.valence?.toFixed(3)}</p>
    <p>Arousal: {audioAnalysis.arousal?.toFixed(3)}</p>
    <p>
      RGB: {audioAnalysis.lighting.rgb.r},{" "}
      {audioAnalysis.lighting.rgb.g},{" "}
      {audioAnalysis.lighting.rgb.b}
    </p>
  </div>
)}


          </div>

          <div className="control-section">
            <label className="section-label">LIGHTING MOOD</label>
            <div className="mood-grid">
              {["CALM", "DREAMY", "ENERGETIC", "INTENSE"].map((item) => (
                <button
                  key={item}
                  className={`mood-button ${mood === item ? "selected" : ""}`}
                  onClick={() => {
                    setMood(item);
                    const colors = {
                      CALM: "#38bdf8",
                      DREAMY: "#a78bfa",
                      ENERGETIC: "#f43f8e",
                      INTENSE: "#ff6b35",
                    };
                    setColor(colors[item]);
                  }}
                >
                  <span
                    className="mood-dot"
                    style={{
                      background: {
                        CALM: "#38bdf8",
                        DREAMY: "#a78bfa",
                        ENERGETIC: "#f43f8e",
                        INTENSE: "#ff6b35",
                      }[item],
                    }}
                  />
                  {item}
                </button>
              ))}
            </div>
          </div>

          <div className="control-section intensity-section">
            <div className="slider-heading">
              <label className="section-label">LIGHT INTENSITY</label>
              <strong>{intensity}%</strong>
            </div>
            <input
              className="intensity-slider"
              type="range"
              min="10"
              max="100"
              value={intensity}
              onChange={(event) => setIntensity(Number(event.target.value))}
              style={{ "--intensity-color": color }}
            />
            <div className="slider-labels">
              <span>SUBTLE</span>
              <span>POWERFUL</span>
            </div>
          </div>

          <button
  className="launch-button"
  onClick={togglePlayback}
  disabled={!track}
>
  {playing ? "Ⅱ  PAUSE SONG & VISUALS" : "▶  PLAY SONG & VISUALS"}
</button>



<p className="sidebar-footnote">
  Python API: {apiStatus}
</p>


          <p className="sidebar-footnote">
            Preview mode · AI integration coming next
          </p>
        </aside>

        <section className="stage-column">
          <div className="stage-heading">
            <div>
              <p className="eyebrow">LIGHTHIVE / ENVIRONMENT 01</p>
              <h2>The main stage<span>.</span></h2>
            </div>
            <div className="stage-live">
              <span className="status-dot" />
              {playing ? "RUNNING" : "STANDBY"}
            </div>
          </div>

         
<div
  className={`concert-stage ${playing ? "is-playing" : ""} ${
    beatPulse ? "beat-pulse" : ""
  } ${climaxActive ? "gesture-climax" : ""} ${
    fullRevealActive ? "gesture-full-reveal" : ""
  } ${autoLightingPaused ? "auto-lighting-paused" : ""}`}
  style={{
    "--stage-color": color,
    "--stage-intensity": intensity / 100,
    "--spotlight-x": `${spotlightDirection?.x ?? 0.5}`,
    "--spotlight-y": `${spotlightDirection?.y ?? 0.5}`,
  }}
>

            <div className="stage-grid" />
            <div className="stage-stars">
              {Array.from({ length: 35 }, (_, i) => (
                <i
                  key={i}
                  style={{
                    left: `${(i * 37 + 11) % 100}%`,
                    top: `${(i * 23 + 7) % 65}%`,
                    animationDelay: `${(i % 8) * 0.3}s`,
                  }}
                />
              ))}
            </div>

            <div className="stage-caption">
              <span>LIVE VISUALIZER</span>
              <span>01 — 01</span>
            </div>

            <div className="light-rig">
              {Array.from({ length: 7 }, (_, i) => (
                <div className="rig-light" key={i}>
                  <div className="rig-head" />
                  <div className={`light-beam beam-${i + 1}`} />
                </div>
              ))}
            </div>

            <div className="ambient-glow" />

<div
  className="performer"
  style={{
    "--left-arm-angle": poseLandmarks
      ? `${Math.atan2(
          poseLandmarks.left_wrist.y -
            poseLandmarks.left_shoulder.y,
          poseLandmarks.left_wrist.x -
            poseLandmarks.left_shoulder.x
        ) * (180 / Math.PI) - 90}deg`
      : "22deg",

    "--right-arm-angle": poseLandmarks
      ? `${Math.atan2(
          poseLandmarks.right_wrist.y -
            poseLandmarks.right_shoulder.y,
          poseLandmarks.right_wrist.x -
            poseLandmarks.right_shoulder.x
        ) * (180 / Math.PI) - 90}deg`
      : "-25deg",
  }}
>
  <div className="performer-head" />
  <div className="performer-body" />
  <div className="performer-arm left-arm" />
  <div className="performer-arm right-arm" />
  <div className="performer-leg left-leg" />
  <div className="performer-leg right-leg" />
</div>
            <div className="platform platform-back" />
            <div className="platform platform-middle" />
            <div className="platform platform-front" />

            <div className="stage-floor-glow" />
            <div className="stage-bottom">
              <span>VALENCE / VISUAL MOOD</span>
              <strong>{mood}</strong>
            </div>

            <div className="stage-center-label">
              <span className="center-line" />
              <span>LIGHTHIVE</span>
              <span className="center-line" />
            </div>
          </div>
           <div className="camera-panel">
  <div className="camera-heading">
    <div>
      <p className="eyebrow">CAMERA ANALYSIS</p>
      <h3>Choose your mode</h3>
    </div>
    <span className="camera-status">{cameraStatus}</span>
  </div>

  <div className="camera-modes">
    <button
      className={`camera-mode ${
        cameraMode === "performer" ? "selected" : ""
      }`}
      onClick={() => setCameraMode("performer")}
    >
      Performer Gesture
    </button>

    <button
      className={`camera-mode ${
        cameraMode === "audience" ? "selected" : ""
      }`}
      onClick={() => setCameraMode("audience")}
    >
      Audience Analysis
    </button>
  </div>

  <div className="camera-preview">
    <video ref={videoRef} autoPlay muted playsInline />
    {!cameraStreamRef.current && (
      <p>Camera preview will appear here</p>
    )}
  </div>

  <div className="camera-actions">
    <button className="camera-action" onClick={startCamera}>
      Start Camera
    </button>
    <button className="camera-action" onClick={stopCamera}>
      Stop Camera
    </button>
  </div>
  
<div className="camera-actions">
  <button
    className="camera-action"
    disabled={cameraStatus !== "Camera connected"}
    onClick={() => {
      setGestureEnabled((enabled) => !enabled);
      setGestureStatus(
        gestureEnabled ? "Detection stopped" : "Starting detection..."
      );
    }}
  >
    {gestureEnabled ? "Stop Detection" : "Start Detection"}
  </button>
</div>

<p className="camera-description">
  Detection status: {gestureStatus}
</p>

<p className="camera-description">
  Pose tracking: {poseStatus}
</p>


<p className="camera-description">
  Latest gesture: {detectedGesture?.event || "None detected yet"}
</p>
<p className="camera-description">
  Lighting action: {gestureAction}
</p>

{detectedGesture?.direction && (
  <p className="camera-description">
    Spotlight direction: X {detectedGesture.direction.x}, Y{" "}
    {detectedGesture.direction.y}
  </p>
)}


 
<p className="camera-description">
  {cameraMode === "performer"
    ? "Performer mode selected."
    : `Audience analysis: ${audienceStatus}`}
</p>

  {cameraError && <p className="camera-error">{cameraError}</p>}
</div>
          <div className="metrics-grid">
            <article className="metric-card">
              <span className="metric-label">LIGHTING ENGINE</span>
             
<strong className="metric-value">
  {apiStatus === "Connected"
    ? "CONNECTED"
    : apiStatus.toUpperCase()}
</strong>

              <span className="metric-note">Local preview</span>
            </article>

            <article className="metric-card">
              <span className="metric-label">COLOR OUTPUT</span>
          

<strong className="metric-value color-value">
  {apiState?.rgb
    ? `RGB ${apiState.rgb.r}, ${apiState.rgb.g}, ${apiState.rgb.b}`
    : color}
</strong>


              <span className="metric-note">RGB visual simulation</span>
            </article>

<article className="metric-card">
  <span className="metric-label">AUDIENCE ENERGY</span>

  <strong className="metric-value">
    {audienceData
      ? Math.round(audienceData.audience_energy)
      : "—"}
  </strong>

  <span className="metric-note">
    {audienceData
      ? `${audienceData.activity_level} activity · ${audienceData.person_count} people`
      : audienceStatus}
  </span>
</article>


            <article className="metric-card">
              <span className="metric-label">BEAT SYNC</span>
              <strong className="metric-value">
                {playing ? "PREVIEW" : "STANDBY"}
              </strong>
              <span className="metric-note">Engine integration pending</span>
            </article>
          </div>

          <footer className="stage-footer">
            <span>LIGHTHIVE © 2026</span>
            <span>INTELLIGENT STAGE ENVIRONMENT</span>
          </footer>
        </section>
      </section>
    </main>
  );
}
