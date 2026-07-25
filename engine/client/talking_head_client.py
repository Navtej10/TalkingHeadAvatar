import time
import httpx
from typing import Optional, Dict, Any, Union

class TalkingHeadClientError(Exception):
    pass

class TalkingHeadClient:
    """
    A strictly typed client library for the Talking-Head Engine API.
    """
    def __init__(self, base_url: str = "http://localhost:8000", api_key: str = None):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key

    def submit(
        self, 
        image_path: Optional[str] = None, 
        identity_name: Optional[str] = None,
        audio_path: Optional[str] = None, 
        text: Optional[str] = None,
        emotion: str = "neutral",
        gaze_target: str = "camera"
    ) -> str:
        """
        Submits a new generation job to the engine.
        Returns the unique job_id.
        """
        url = f"{self.base_url}/jobs"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        data = {
            "emotion": emotion,
            "gaze_target": gaze_target
        }
        
        if text:
            data["text"] = text
        if identity_name:
            data["identity_name"] = identity_name

        files = {}
        file_handles = []
        
        try:
            if image_path:
                f_img = open(image_path, "rb")
                file_handles.append(f_img)
                files["image"] = (image_path.split("/")[-1], f_img, "image/jpeg")
                
            if audio_path:
                f_audio = open(audio_path, "rb")
                file_handles.append(f_audio)
                files["audio"] = (audio_path.split("/")[-1], f_audio, "audio/wav")

            response = httpx.post(url, headers=headers, data=data, files=files if files else None, timeout=30.0)
            response.raise_for_status()
            
            resp_data = response.json()
            return resp_data.get("job_id")
            
        except httpx.HTTPStatusError as e:
            raise TalkingHeadClientError(f"API Error ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            raise TalkingHeadClientError(f"Request failed: {e}") from e
        finally:
            for f in file_handles:
                f.close()

    def poll(self, job_id: str) -> Dict[str, Any]:
        """
        Polls the current status of a given job.
        Returns a dictionary containing 'status', 'result_url', and 'error'.
        """
        url = f"{self.base_url}/jobs/{job_id}"
        headers = {}
        if self.api_key:
            headers["X-API-Key"] = self.api_key

        try:
            response = httpx.get(url, headers=headers, timeout=10.0)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            raise TalkingHeadClientError(f"API Error ({e.response.status_code}): {e.response.text}") from e
        except Exception as e:
            raise TalkingHeadClientError(f"Polling failed: {e}") from e

    def wait(self, job_id: str, timeout: float = 300.0, poll_interval: float = 2.0) -> str:
        """
        Blocks and continuously polls until the job completes or fails.
        Returns the result_path (URL) on success.
        Raises TalkingHeadClientError on failure or timeout.
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            status_data = self.poll(job_id)
            status = status_data.get("status")
            
            if status == "done":
                return status_data.get("result_url")
            elif status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                raise TalkingHeadClientError(f"Job failed: {error_msg}")
            elif status == "not_found":
                raise TalkingHeadClientError(f"Job {job_id} not found on server.")
                
            time.sleep(poll_interval)
            
        raise TalkingHeadClientError(f"Timeout exceeded waiting for job {job_id}")
