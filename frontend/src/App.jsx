import { useWebRTC } from "./hooks/useWebRTC.js";
import AudioRecorder from "./components/AudioRecorder.jsx";
import WaveformVisualizer from "./components/WaveformVisualizer.jsx";
import ConversationLog from "./components/ConversationLog.jsx";

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || "";

export default function App() {
  const {
    connect,
    disconnect,
    connected,
    status,
    transcript,
    aiText,
    emotion,
    aiSpeaking,
    remoteAudioRef,
    localStream,
  } = useWebRTC({ backendUrl: BACKEND_URL });

  return (
    <div
      style={{
        maxWidth: 720,
        margin: "0 auto",
        padding: 24,
        fontFamily: "Inter, system-ui, sans-serif",
        color: "white",
        background: "#171923",
        minHeight: "100vh",
      }}
    >
      <h1 style={{ marginBottom: 4 }}>Auralis</h1>
      <p style={{ opacity: 0.7, marginTop: 0 }}>Crisis Negotiation Training Simulator</p>

<div
  style={{
    marginTop: 16,
    padding: 14,
    borderRadius: 10,
    background: "#202331",
    border: "1px solid #303545",
  }}
>
  <div style={{ fontSize: 13, fontWeight: 600, marginBottom: 8 }}>
    Live Session Status
  </div>

  <div style={{ display: "flex", gap: 18, flexWrap: "wrap", fontSize: 13 }}>
    <span>
      Connection: <strong>{connected ? "Connected" : "Disconnected"}</strong>
    </span>

    <span>
      AI: <strong>{aiSpeaking ? "Speaking" : "Listening"}</strong>
    </span>

    <span>
  Emotion:{" "}
  <strong>
    {emotion?.emotion
      ? `${emotion.emotion} (${emotion.arousal?.toFixed?.(2) ?? "N/A"})`
      : "Waiting"}
  </strong>
</span>
  </div>

  <div style={{ marginTop: 8, fontSize: 12, opacity: 0.6 }}>
    {status || "Ready to start a voice session"}
  </div>
</div>

      <AudioRecorder
        connected={connected}
        status={status}
        onConnect={connect}
        onDisconnect={disconnect}
        aiSpeaking={aiSpeaking}
      />

      <div style={{ marginTop: 20, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 12 }}>
        <div>
          <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>You</div>
          <WaveformVisualizer
            mediaStream={connected ? localStream.current : null}
            role="user"
            arousal={emotion?.arousal ?? 0}
          />
        </div>
        <div>
          <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>Auralis</div>
          <WaveformVisualizer
            audioEl={connected ? remoteAudioRef.current : null}
            role="ai"
          />
        </div>
      </div>

      {/* Hidden audio element that actually plays the AI's streamed response */}
      <audio ref={remoteAudioRef} autoPlay style={{ display: "none" }} />

      <ConversationLog transcript={transcript} emotion={emotion} aiText={aiText} />
    </div>
  );
}

