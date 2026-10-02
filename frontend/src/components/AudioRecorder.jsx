export default function AudioRecorder({
  connected,
  status,
  onConnect,
  onDisconnect,
  aiSpeaking,
}) {
  const sessionMessage =
    status === "connecting"
      ? "Connecting to Auralis..."
      : connected
        ? aiSpeaking
          ? "Auralis is speaking - you can interrupt by talking"
          : "Listening for your voice..."
        : status === "error"
          ? "Connection failed - please try again"
          : "Ready to start a voice session";

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
      <button
        onClick={connected ? onDisconnect : onConnect}
        disabled={status === "connecting"}
        style={{
          padding: "10px 20px",
          borderRadius: 8,
          border: "none",
          fontWeight: 600,
          cursor: "pointer",
          background: connected ? "#e53e3e" : "#38a169",
          color: "white",
        }}
      >
        {status === "connecting"
          ? "Connecting..."
          : connected
            ? "Disconnect"
            : "Connect"}
      </button>

      <span style={{ fontSize: 14, opacity: 0.8 }}>
        {sessionMessage}
      </span>

      <span
        style={{
          width: 10,
          height: 10,
          borderRadius: 8,
          background:
            status === "error"
              ? "#e53e3e"
              : connected
                ? "#38a169"
                : "#a0aec0",
          display: "inline-block",
        }}
      />
    </div>
  );
}
