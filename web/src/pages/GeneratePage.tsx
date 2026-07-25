import { useState } from "react";
import { createJob, getJobStatus, JobStatusResponse } from "../api/client";

export default function GeneratePage() {
  const [image, setImage] = useState<File | null>(null);
  const [text, setText] = useState("");
  const [status, setStatus] = useState<JobStatusResponse | null>(null);

  async function handleSubmit() {
    if (!image) return;
    const job = await createJob(image, { text });
    setStatus({ job_id: job.job_id, status: job.status });
    poll(job.job_id);
  }

  function poll(jobId: string) {
    const interval = setInterval(async () => {
      const s = await getJobStatus(jobId);
      setStatus(s);
      if (s.status === "done" || s.status === "failed") clearInterval(interval);
    }, 2000);
  }

  return (
    <div style={{ maxWidth: 480, margin: "40px auto", fontFamily: "sans-serif" }}>
      <h1>Talking-Head Engine</h1>
      <input type="file" accept="image/*" onChange={(e) => setImage(e.target.files?.[0] ?? null)} />
      <textarea
        placeholder="Text to speak (or attach audio instead)"
        value={text}
        onChange={(e) => setText(e.target.value)}
        style={{ width: "100%", marginTop: 12, minHeight: 80 }}
      />
      <button onClick={handleSubmit} style={{ marginTop: 12 }}>
        Generate
      </button>

      {status && (
        <div style={{ marginTop: 24 }}>
          <p>Status: {status.status}</p>
          {status.result_url && (
            <video src={status.result_url} controls style={{ width: "100%" }} />
          )}
          {status.error && <p style={{ color: "red" }}>{status.error}</p>}
        </div>
      )}
    </div>
  );
}
