import { useCallback, useRef, useState } from "react";

/**
 * useWebRTC
 *
 * Establishes a direct WebRTC PeerConnection with the Auralis backend
 * (bypassing slow HTTP request/response cycles per-utterance — this is the
 * Week 1 foundation). Handles:
 *   - capturing the mic and sending it as an outgoing audio track
 *   - receiving the AI's streamed response audio track and playing it
 *   - an RTCDataChannel carrying JSON events (live transcript, emotion,
 *     AI sentences, interruption/barge-in notifications) for the UI
 */
export function useWebRTC({ backendUrl = "" } = {}) {
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState("idle"); // idle | connecting | connected | error
  const [transcript, setTranscript] = useState("");
  const [aiText, setAiText] = useState("");
  const [emotion, setEmotion] = useState(null);
  const [aiSpeaking, setAiSpeaking] = useState(false);

  const pcRef = useRef(null);
  const dcRef = useRef(null);
  const localStreamRef = useRef(null);
  const remoteAudioRef = useRef(null);
  const sessionIdRef = useRef(null);

  const connect = useCallback(async () => {
    setStatus("connecting");
    try {
      const pc = new RTCPeerConnection({
        iceServers: [{ urls: "stun:stun.l.google.com:19302" }],
      });
      pcRef.current = pc;

      // Outgoing: capture mic and add as a track
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });
      localStreamRef.current = stream;
      stream.getTracks().forEach((track) => pc.addTrack(track, stream));

      // Incoming: play the AI's response audio track as it streams in
      pc.ontrack = (event) => {
        if (remoteAudioRef.current) {
          remoteAudioRef.current.srcObject = event.streams[0];
          remoteAudioRef.current.play().catch(() => {});
        }
      };

      // Data channel for transcript / emotion / AI text events
      const dc = pc.createDataChannel("events");
      dcRef.current = dc;
      dc.onmessage = (msg) => {
        const event = JSON.parse(msg.data);
        switch (event.type) {
          case "user_turn":
            setTranscript(event.transcript);
            setEmotion(event.emotion);
            break;
          case "ai_sentence":
            setAiText((prev) => (prev ? prev + " " + event.text : event.text));
            setAiSpeaking(true);
            break;
          case "ai_turn_complete":
            setAiSpeaking(false);
            break;
          case "interruption":
            setAiSpeaking(false);
            setAiText("");
            break;
          default:
            break;
        }
      };

      const offer = await pc.createOffer();
      await pc.setLocalDescription(offer);

      const response = await fetch(`${backendUrl}/offer`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ sdp: offer.sdp, type: offer.type }),
      });
      const answer = await response.json();
      sessionIdRef.current = answer.session_id;
      await pc.setRemoteDescription(answer);

      setConnected(true);
      setStatus("connected");
    } catch (err) {
      console.error("WebRTC connection failed:", err);
      setStatus("error");
    }
  }, [backendUrl]);

  const disconnect = useCallback(async () => {
    if (sessionIdRef.current) {
      await fetch(`${backendUrl}/session/${sessionIdRef.current}/close`, {
        method: "POST",
      }).catch(() => {});
    }
    localStreamRef.current?.getTracks().forEach((t) => t.stop());
    pcRef.current?.close();
    setConnected(false);
    setStatus("idle");
    setTranscript("");
    setAiText("");
    setEmotion(null);
    setAiSpeaking(false);
  }, [backendUrl]);

  return {
    connect,
    disconnect,
    connected,
    status,
    transcript,
    aiText,
    emotion,
    aiSpeaking,
    remoteAudioRef,
    localStream: localStreamRef,
  };
}
