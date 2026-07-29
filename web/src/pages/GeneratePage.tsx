import { useState, useRef, useEffect } from "react";
import { createJob, getJobStatus, JobStatusResponse } from "../api/client";

export default function GeneratePage() {
  const [image, setImage] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [audioFile, setAudioFile] = useState<File | null>(null);
  const [status, setStatus] = useState<JobStatusResponse | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);

  // Recording state
  const [isRecording, setIsRecording] = useState(false);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<BlobPart[]>([]);

  // Keep a single polling interval — cleared before creating a new one
  const pollIntervalRef = useRef<ReturnType<typeof setInterval> | null>(null);

  // Clean up interval on unmount
  useEffect(() => {
    return () => {
      if (pollIntervalRef.current !== null) {
        clearInterval(pollIntervalRef.current);
      }
    };
  }, []);

  const isRunning = status?.status === "queued" || status?.status === "running";

  async function handleSubmit() {
    if (!image) return;
    if (isRunning || isSubmitting) return; // Prevent duplicate submissions

    // Clear any existing polling interval before starting a new job
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    setIsSubmitting(true);
    try {
      const job = await createJob(image, {
        text: text ? text : undefined,
        audio: audioFile ? audioFile : undefined,
      });
      setStatus({ job_id: job.job_id, status: job.status });
      poll(job.job_id);
    } catch (err) {
      setStatus({ job_id: "", status: "failed", error: String(err) });
    } finally {
      setIsSubmitting(false);
    }
  }

  function poll(jobId: string) {
    // Guarantee only one interval is alive at a time
    if (pollIntervalRef.current !== null) {
      clearInterval(pollIntervalRef.current);
    }
    pollIntervalRef.current = setInterval(async () => {
      try {
        const s = await getJobStatus(jobId);
        setStatus(s);
        if (s.status === "done" || s.status === "failed") {
          clearInterval(pollIntervalRef.current!);
          pollIntervalRef.current = null;
        }
      } catch {
        // Network error — keep polling
      }
    }, 3000); // Poll every 3 s (was 2 s — reduces server load)
  }

  async function startRecording() {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
      const mediaRecorder = new MediaRecorder(stream);
      mediaRecorderRef.current = mediaRecorder;
      audioChunksRef.current = [];

      mediaRecorder.ondataavailable = (event) => {
        if (event.data.size > 0) {
          audioChunksRef.current.push(event.data);
        }
      };

      mediaRecorder.onstop = () => {
        const audioBlob = new Blob(audioChunksRef.current, { type: "audio/webm" });
        const file = new File([audioBlob], "recording.webm", { type: "audio/webm" });
        setAudioFile(file);
        stream.getTracks().forEach((track) => track.stop());
      };

      mediaRecorder.start();
      setIsRecording(true);
    } catch (error) {
      console.error("Error accessing microphone:", error);
      alert("Could not access microphone.");
    }
  }

  function stopRecording() {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
    }
  }

  const submitDisabled = !image || (!audioFile && !text) || isRunning || isSubmitting;

  const statusLabel: Record<string, string> = {
    queued: "Queued…",
    running: "Processing…",
    done: "Done",
    failed: "Failed",
  };

  return (
    <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Talking-Head Engine</h1>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>1. Choose Image</label>
        <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
      </div>

      <div style={{ marginBottom: 12 }}>
        <label style={{ display: "block", marginBottom: 4, fontWeight: "bold" }}>2. Choose Audio OR Text</label>

        <div style={{ padding: 12, border: "1px solid #ccc", borderRadius: 4, marginBottom: 8, background: audioFile ? "#f9f9f9" : "transparent" }}>
          <p style={{ margin: "0 0 8px 0", fontWeight: "bold" }}>Audio Input</p>
          <input
            type="file"
            accept="audio/*"
            disabled={text.length > 0}
            onChange={(e) => setAudioFile(e.target.files?.[0] ?? null)}
            style={{ marginBottom: 8, display: "block" }}
          />

          <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
            <button
              onClick={isRecording ? stopRecording : startRecording}
              disabled={text.length > 0}
              style={{
                background: isRecording ? "#ff4444" : "#4CAF50",
                color: "white",
                border: "none",
                padding: "6px 12px",
                borderRadius: 4,
                cursor: text.length > 0 ? "not-allowed" : "pointer",
              }}
            >
              {isRecording ? "Stop Recording" : "Record Voice"}
            </button>
            {isRecording && <span style={{ color: "red", fontSize: "14px", animation: "blink 1s infinite" }}>Recording…</span>}
            {audioFile && <span style={{ fontSize: "14px", color: "green" }}>✓ Audio ready</span>}
          </div>
          {audioFile && (
            <button onClick={() => setAudioFile(null)} style={{ marginTop: 8, fontSize: "12px" }}>
              Clear Audio
            </button>
          )}
        </div>

        <div style={{ padding: 12, border: "1px solid #ccc", borderRadius: 4, background: text.length > 0 ? "#f9f9f9" : "transparent" }}>
          <p style={{ margin: "0 0 8px 0", fontWeight: "bold" }}>Text-to-Speech Fallback</p>
          <textarea
            placeholder="Text to speak…"
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={audioFile !== null}
            style={{ width: "100%", boxSizing: "border-box", minHeight: 80, opacity: audioFile !== null ? 0.5 : 1 }}
          />
        </div>
      </div>

      <button
        onClick={handleSubmit}
        disabled={submitDisabled}
        style={{
          marginTop: 12,
          padding: "10px 20px",
          fontSize: "16px",
          background: submitDisabled ? "#ccc" : "#007bff",
          color: "white",
          border: "none",
          borderRadius: 4,
          cursor: submitDisabled ? "not-allowed" : "pointer",
        }}
      >
        {isRunning ? "Processing…" : isSubmitting ? "Submitting…" : "Generate"}
      </button>

      {status && (
        <div style={{ marginTop: 24, padding: 12, border: "1px solid #eee", borderRadius: 4 }}>
          <p>
            Status: <strong>{statusLabel[status.status] ?? status.status}</strong>
            {status.job_id && (
              <span style={{ marginLeft: 8, fontSize: "12px", color: "#888" }}>
                #{status.job_id}
              </span>
            )}
          </p>
          {isRunning && (
            <div style={{ height: 4, background: "#eee", borderRadius: 2, marginTop: 8 }}>
              <div
                style={{
                  height: "100%",
                  background: "#007bff",
                  borderRadius: 2,
                  width: "40%",
                  animation: "indeterminate 1.5s infinite ease-in-out",
                }}
              />
            </div>
          )}
          {status.result_url && (
            <div style={{ marginTop: 12 }}>
              <video src={status.result_url} controls style={{ width: "100%", borderRadius: 8, background: "black" }} />
            </div>
          )}
          {status.error && <p style={{ color: "red", marginTop: 8 }}>{status.error}</p>}
        </div>
      )}
      <style>{`
        @keyframes blink {
          0% { opacity: 1; }
          50% { opacity: 0; }
          100% { opacity: 1; }
        }
        @keyframes indeterminate {
          0%   { transform: translateX(-100%); width: 40%; }
          50%  { width: 60%; }
          100% { transform: translateX(350%); width: 40%; }
        }
      `}</style>
    </div>
  );
}
