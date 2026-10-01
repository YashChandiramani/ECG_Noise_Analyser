import React, { useState, useMemo } from "react";
import {
  Activity, AlertTriangle, CheckCircle2, Cpu, FileHeart,
  FileUp, Gauge, Info, LoaderCircle, RotateCcw, ShieldCheck,
  SlidersHorizontal, Sparkles, UploadCloud, Waves, X
} from "lucide-react";
import { Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

const API_URL = import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

function syntheticPreview() {
  return Array.from({ length: 180 }, (_, i) => {
    const t = i / 36;
    const value =
      0.12 * Math.sin(t * 5.1) +
      0.05 * Math.sin(t * 17) +
      (i % 45 === 18 ? 0.75 : 0) +
      (i % 45 === 20 ? -0.28 : 0) +
      (i % 45 === 22 ? 0.35 : 0);
    return { x: i, value };
  });
}

function parseCsvText(text) {
  const values = text
    .split(/[\s,;\n\r]+/)
    .map(v => v.trim())
    .filter(Boolean)
    .map(Number)
    .filter(Number.isFinite);

  return values;
}

function App() {
  const [file, setFile] = useState(null);
  const [signal, setSignal] = useState([]);
  const [model, setModel] = useState("1d_cnn");
  const [result, setResult] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  const preview = useMemo(() => {
    if (!signal.length) return syntheticPreview();
    const step = Math.max(1, Math.floor(signal.length / 180));
    return signal.filter((_, i) => i % step === 0).slice(0, 180).map((value, i) => ({ x: i, value }));
  }, [signal]);

  const chooseFile = async (selected) => {
    setError("");
    setResult(null);
    setFile(selected || null);
    if (!selected) {
      setSignal([]);
      return;
    }

    const text = await selected.text();
    const values = parseCsvText(text);
    if (!values.length) {
      setError("No numeric ECG samples were found in this file.");
      setSignal([]);
      return;
    }
    setSignal(values);
  };

  const analyze = async () => {
    if (!file) {
      setError("Please upload an ECG CSV file first.");
      return;
    }

    setBusy(true);
    setError("");
    setResult(null);

    try {
      const body = new FormData();
      body.append("file", file);
      body.append("model", model);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body
      });

      if (!response.ok) {
        let detail = "Prediction request failed.";
        try {
          const errData = await response.json();
          detail = errData.detail || errData.message || JSON.stringify(errData);
        } catch {
          detail = await response.text();
        }
        throw new Error(detail || "Prediction request failed.");
      }

      setResult(await response.json());
    } catch (err) {
      console.error(err);

      if (err.name === "TypeError" && err.message?.includes("Failed to fetch")) {
        setError(
          "Unable to connect to the backend server. If using Render's free tier, the server spins down after inactivity and takes ~40-50 seconds to wake up on the first request. Please wait a moment and try again."
        );
      } else {
        setError(
          err.message ||
          "Unable to analyze the ECG signal. Please make sure the FastAPI server is running."
        );
      }
    } finally {
      setBusy(false);
    }
  };

  const reset = () => {
    setFile(null);
    setSignal([]);
    setResult(null);
    setError("");
  };

  const isClean = result?.prediction?.class_id === 0;

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark"><Activity size={22} /></div>
          <div>
            <div className="brand-name">ECG<span>Sense</span></div>
            <div className="brand-sub">Clinical Signal Intelligence</div>
          </div>
        </div>
        <div className="system-status"><span /> Model workspace ready</div>
      </header>

      <main className="page">
        <section className="hero">
          <div>
            <div className="eyebrow"><Sparkles size={15} /> AI-assisted ECG analysis</div>
            <h1>Detect clinical noise<br /><em>before it hides the signal.</em></h1>
            <p>
              Upload an ECG segment and inspect its waveform before sending it
              through the selected deep-learning branch.
            </p>
          </div>
          <div className="hero-card">
            <Waves size={26} />
            <div><strong>5 s analysis window</strong><small>Notebook target: resampled ECG signal</small></div>
          </div>
        </section>

        <section className="workspace">
          <div className="panel upload-panel">
            <div className="panel-head">
              <div>
                <div className="section-kicker">01 · INPUT</div>
                <h2>Upload ECG signal</h2>
              </div>
              {file && <button className="icon-btn" onClick={reset} title="Reset"><RotateCcw size={17}/></button>}
            </div>

            <label
              className={`dropzone ${file ? "has-file" : ""}`}
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => { e.preventDefault(); chooseFile(e.dataTransfer.files?.[0]); }}
            >
              <input
                type="file"
                accept=".csv,.txt"
                onChange={(e) => chooseFile(e.target.files?.[0])}
              />
              {file ? (
                <>
                  <div className="file-icon"><FileHeart size={25}/></div>
                  <strong>{file.name}</strong>
                  <span>{signal.length.toLocaleString()} numeric samples detected</span>
                  <small>Click to replace file</small>
                </>
              ) : (
                <>
                  <div className="upload-icon"><UploadCloud size={28}/></div>
                  <strong>Drop your ECG CSV here</strong>
                  <span>or click to browse your computer</span>
                  <small>CSV / TXT · numeric samples only</small>
                </>
              )}
            </label>

            <div className="input-note">
              <Info size={16}/>
              <span>The notebook's raw 1D branch expects an ECG signal vector. Keep preprocessing identical to training when deploying weights.</span>
            </div>
          </div>

          <div className="panel settings-panel">
            <div className="panel-head">
              <div>
                <div className="section-kicker">02 · MODEL</div>
                <h2>Analysis branch</h2>
              </div>
              <SlidersHorizontal size={20} className="muted" />
            </div>

            <div className="model-list">
              <button className={`model-option ${model === "1d_cnn" ? "selected" : ""}`} onClick={() => setModel("1d_cnn")}>
                <span className="model-symbol"><Waves size={19}/></span>
                <span><strong>1D CNN</strong><small>Raw temporal ECG · fast inference</small></span>
                <span className="radio">{model === "1d_cnn" && <i/>}</span>
              </button>

              <button className={`model-option ${model === "resnet18" ? "selected" : ""}`} onClick={() => setModel("resnet18")}>
                <span className="model-symbol"><Cpu size={19}/></span>
                <span><strong>CWT + ResNet-18</strong><small>Time-frequency scalogram branch</small></span>
                <span className="radio">{model === "resnet18" && <i/>}</span>
              </button>
            </div>

            <div className="spec-grid">
              <div><span>Input</span><strong>{model === "1d_cnn" ? "1D signal" : "224 × 224 RGB"}</strong></div>
              <div><span>Output</span><strong>2 classes</strong></div>
              <div><span>Task</span><strong>Clinical noise</strong></div>
              <div><span>Framework</span><strong>PyTorch</strong></div>
            </div>

            <button className="analyze-btn" disabled={busy || !file} onClick={analyze}>
              {busy ? <><LoaderCircle className="spin" size={19}/> Analyzing signal…</> : <><ShieldCheck size={19}/> Analyze ECG</>}
            </button>
          </div>
        </section>

        {error && <div className="error-box"><AlertTriangle size={18}/><span>{error}</span><button onClick={() => setError("")}><X size={16}/></button></div>}

        <section className="panel waveform-panel">
          <div className="panel-head">
            <div>
              <div className="section-kicker">03 · SIGNAL VIEW</div>
              <h2>ECG waveform</h2>
            </div>
            <div className="signal-badge"><span /> {signal.length ? "Uploaded signal" : "Preview"}</div>
          </div>

          <div className="chart-wrap">
            <ResponsiveContainer width="100%" height={310}>
              <LineChart data={preview} margin={{ top: 18, right: 18, left: -22, bottom: 8 }}>
                <XAxis dataKey="x" hide />
                <YAxis domain={["auto", "auto"]} tick={{ fontSize: 11 }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={{ borderRadius: 12, border: "1px solid #e6e8ee", fontSize: 12 }} />
                <Line type="monotone" dataKey="value" dot={false} strokeWidth={2.1} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </section>

        {result && (
  <section className={`result-card ${isClean ? "clean" : "noisy"}`}>
    <div className="result-icon">
      {isClean ? (
        <CheckCircle2 size={34} />
      ) : (
        <AlertTriangle size={34} />
      )}
    </div>

    <div className="result-main">
      <div className="section-kicker">
        04 · CLASSIFICATION
      </div>

      <h2>
        {isClean
          ? "Clean ECG"
          : "Clinical Noise Detected"}
      </h2>

      <p>
  {result.message ||
    `Classification returned by the ${
      result.model === "resnet18"
        ? "CWT + ResNet-18"
        : "1D CNN"
    } model.`}
</p>

      <div className="prediction-details">

        <div className="prediction-stat">
  <span>Model</span>
  <strong>
    {result.model === "resnet18"
      ? "CWT + ResNet-18"
      : "1D CNN"}
  </strong>
</div>

        <div className="prediction-stat">
          <span>Noisy probability</span>
          <strong>
            {(
              (result.probabilities?.noisy ?? 0) * 100
            ).toFixed(2)}
            %
          </strong>
        </div>

        <div className="prediction-stat">
          <span>Samples received</span>
          <strong>
            {result.signal?.samples_received?.toLocaleString() ?? "-"}
          </strong>
        </div>

        <div className="prediction-stat">
          <span>Samples analyzed</span>
          <strong>
            {result.signal?.samples_used?.toLocaleString() ?? "-"}
          </strong>
        </div>

      </div>
    </div>

    <div className="confidence">
      <span>Confidence</span>

      <strong>
        {(
          (result.prediction?.confidence ?? 0) * 100
        ).toFixed(1)}
        %
      </strong>
    </div>
  </section>
)}

        <section className="footer-note">
          <Gauge size={17}/>
          <span>This interface is a research/demo visualization and is not a medical diagnosis.</span>
        </section>
      </main>
    </div>
  );
}

export default App;