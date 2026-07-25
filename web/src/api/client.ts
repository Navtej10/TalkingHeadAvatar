export interface JobResponse {
  job_id: string;
  status: string;
}

export interface JobStatusResponse {
  job_id: string;
  status: string;
  result_url?: string;
  error?: string;
}

export async function createJob(image: File, audioOrText: { audio?: File; text?: string }): Promise<JobResponse> {
  const form = new FormData();
  form.append("image", image);
  if (audioOrText.audio) form.append("audio", audioOrText.audio);
  if (audioOrText.text) form.append("text", audioOrText.text);

  const res = await fetch("/jobs", { 
    method: "POST", 
    body: form,
    headers: {
      "X-API-Key": "test_dev_key"
    }
  });
  if (!res.ok) throw new Error(`Job creation failed: ${res.status}`);
  return res.json();
}

export async function getJobStatus(jobId: string): Promise<JobStatusResponse> {
  const res = await fetch(`/jobs/${jobId}`, {
    headers: {
      "X-API-Key": "test_dev_key"
    }
  });
  if (!res.ok) throw new Error(`Status check failed: ${res.status}`);
  return res.json();
}
