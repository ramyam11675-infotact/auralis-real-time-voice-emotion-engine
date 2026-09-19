const EMOTION_COLORS = {
  "panicked/distressed": "#e53e3e",
  "agitated/excited": "#dd6b20",
  "sad/withdrawn": "#805ad5",
  calm: "#38a169",
  neutral: "#718096",
};

export default function ConversationLog({ transcript, emotion, aiText }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 16, marginTop: 16 }}>
      <div style={{ background: "#1a202c", padding: 16, borderRadius: 8 }}>
        <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>TRAINEE (you)</div>
        <div style={{ color: "white", minHeight: 24 }}>{transcript || "…"}</div>
        {emotion && (
          <div style={{ marginTop: 8, display: "flex", gap: 8, alignItems: "center" }}>
            <span
              style={{
                background: EMOTION_COLORS[emotion.label] || "#718096",
                padding: "2px 10px",
                borderRadius: 999,
                fontSize: 12,
                color: "white",
              }}
            >
              {emotion.label}
            </span>
            <span style={{ fontSize: 12, opacity: 0.6 }}>
              arousal {emotion.arousal.toFixed(2)} · valence {emotion.valence.toFixed(2)} · dominance{" "}
              {emotion.dominance.toFixed(2)}
            </span>
          </div>
        )}
      </div>

      <div style={{ background: "#2d3748", padding: 16, borderRadius: 8 }}>
        <div style={{ fontSize: 12, opacity: 0.6, marginBottom: 4 }}>AURALIS (simulated subject)</div>
        <div style={{ color: "white", minHeight: 24 }}>{aiText || "…"}</div>
      </div>
    </div>
  );
}
