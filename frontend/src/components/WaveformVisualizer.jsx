import { useEffect, useRef } from "react";

/**
 * WaveformVisualizer (Week 4 polish)
 *
 * Draws a live waveform for either the human's mic input or the AI's
 * streamed response audio, using the Web Audio API's AnalyserNode.
 * Color reflects role ("user" vs "ai") and, for the user, the current
 * emotional arousal level (calmer = cooler color, panicked = hotter).
 */
export default function WaveformVisualizer({ mediaStream, audioEl, role = "user", arousal = 0 }) {
  const canvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const rafRef = useRef(null);

  useEffect(() => {
    if (!mediaStream && !audioEl) return;

    const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    audioCtxRef.current = audioCtx;

    const analyser = audioCtx.createAnalyser();
    analyser.fftSize = 1024;

    let source;
    if (mediaStream) {
      source = audioCtx.createMediaStreamSource(mediaStream);
    } else if (audioEl) {
      source = audioCtx.createMediaElementSource(audioEl);
      source.connect(audioCtx.destination); // keep audio audible
    }
    source.connect(analyser);

    const bufferLength = analyser.fftSize;
    const dataArray = new Uint8Array(bufferLength);
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    const baseColor = role === "ai" ? [99, 179, 237] : [237, 100 + arousal * 100, 100];

    function draw() {
      rafRef.current = requestAnimationFrame(draw);
      analyser.getByteTimeDomainData(dataArray);

      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.lineWidth = 2;
      ctx.strokeStyle = `rgb(${baseColor[0]}, ${baseColor[1]}, ${baseColor[2]})`;
      ctx.beginPath();

      const sliceWidth = canvas.width / bufferLength;
      let x = 0;
      for (let i = 0; i < bufferLength; i++) {
        const v = dataArray[i] / 128.0;
        const y = (v * canvas.height) / 2;
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
        x += sliceWidth;
      }
      ctx.lineTo(canvas.width, canvas.height / 2);
      ctx.stroke();
    }
    draw();

    return () => {
      cancelAnimationFrame(rafRef.current);
      source.disconnect();
      analyser.disconnect();
      audioCtx.close();
    };
  }, [mediaStream, audioEl, role, arousal]);

  return (
    <canvas
      ref={canvasRef}
      width={480}
      height={80}
      style={{ width: "100%", height: "80px", background: "#111", borderRadius: 8 }}
    />
  );
}
