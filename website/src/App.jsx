
import { useState } from "react";
import "./App.css";

export default function App() {
  const [playing, setPlaying] = useState(false);
  const [intensity, setIntensity] = useState(70);
  const [mood, setMood] = useState("ENERGETIC");
  const [track, setTrack] = useState(null);
  const [color, setColor] = useState("#8b5cf6");
  
const [apiState, setApiState] = useState(null);
const [apiStatus, setApiStatus] = useState("Checking");

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



const handleUpload = async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;

  setTrack(file);
  setAudioStatus("Analysing song...");

  try {
    const formData = new FormData();
    formData.append("file", file);

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
};



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

            {track && (
              <audio
                className="audio-player"
                controls
                src={URL.createObjectURL(track)}
                onPlay={() => setPlaying(true)}
                onPause={() => setPlaying(false)}
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
            onClick={() => setPlaying(!playing)}
          >
            {playing ? "Ⅱ  PAUSE VISUALS" : "▶  LAUNCH VISUALS"}
          </button>
          
<button
  className="launch-button"
  style={{ marginTop: 10, background: "#25233a" }}
  onClick={fetchLightingState}
>
  ↻ SYNC WITH PYTHON
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
            className={`concert-stage ${playing ? "is-playing" : ""}`}
            style={{
              "--stage-color": color,
              "--stage-intensity": intensity / 100,
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

            <div className="performer">
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
              <strong className="metric-value">—</strong>
              <span className="metric-note">Awaiting camera integration</span>
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
