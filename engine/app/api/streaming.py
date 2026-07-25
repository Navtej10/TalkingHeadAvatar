from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json
import uuid
import os
import shutil
import base64
import cv2
import tempfile
from app.config import INPUT_DIR, TMP_DIR
from app.pipeline.orchestrator import STAGES
from app.core.identity_store import load_identity

router = APIRouter(prefix="/stream", tags=["Streaming"])

@router.websocket("")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    
    session_id = uuid.uuid4().hex[:8]
    os.makedirs(f"{TMP_DIR}/{session_id}", exist_ok=True)
    os.makedirs(INPUT_DIR, exist_ok=True)
    
    context = {
        "job_id": session_id,
        "emotion": "neutral",
        "gaze_target": "camera"
    }
    
    try:
        # Step 1: Wait for initial configuration (identity)
        init_message = await websocket.receive_text()
        init_data = json.loads(init_message)
        
        api_key = init_data.get("api_key")
        if not api_key:
            await websocket.send_text(json.dumps({"error": "Missing api_key in init payload"}))
            await websocket.close()
            return
            
        # Verify API Key manually for WebSocket since Depends() is harder here
        from app.db.session import SessionLocal
        from app.db.models import ApiKey
        db = SessionLocal()
        try:
            db_key = db.query(ApiKey).filter(ApiKey.key == api_key).first()
            if not db_key or db_key.credits_remaining <= 0:
                await websocket.send_text(json.dumps({"error": "Invalid API Key or insufficient credits"}))
                await websocket.close()
                return
        finally:
            db.close()
            
        # We don't deduct credits for WS streaming in this implementation (would need a per-chunk or per-minute billing model)
        
        identity_name = init_data.get("identity_name")
        image_b64 = init_data.get("image")
        emotion = init_data.get("emotion", "neutral")
        gaze_target = init_data.get("gaze_target", "camera")
        
        context["emotion"] = emotion
        context["gaze_target"] = gaze_target
        
        if identity_name:
            context["identity"] = load_identity(identity_name)
            context["face"] = {"aligned_face": cv2.imread(f"{INPUT_DIR}/{identity_name}_thumbnail.jpg")}
        elif image_b64:
            image_data = base64.b64decode(image_b64)
            image_path = f"{INPUT_DIR}/{session_id}_image.jpg"
            with open(image_path, "wb") as f:
                f.write(image_data)
            context["image_path"] = image_path
            
            # Run Stage 1 & 2 to establish baseline
            context = STAGES[0]().run(context)
            context = STAGES[1]().run(context)
        else:
            await websocket.send_text(json.dumps({"error": "Must provide identity_name or image_b64"}))
            await websocket.close()
            return
            
        await websocket.send_text(json.dumps({"status": "ready", "session_id": session_id}))
        
        # Step 2: Continuous audio stream loop
        chunk_index = 0
        while True:
            # Receive binary audio chunk
            audio_bytes = await websocket.receive_bytes()
            chunk_index += 1
            
            audio_path = f"{TMP_DIR}/{session_id}/audio_chunk_{chunk_index}.wav"
            with open(audio_path, "wb") as f:
                f.write(audio_bytes)
                
            # Clone context to avoid mutating the baseline for subsequent chunks
            chunk_context = context.copy()
            chunk_context["audio_path"] = audio_path
            
            # Run Stages 3 through 9 (skip Assembly to stream raw frames)
            # (In a highly optimized pipeline, you'd keep models loaded in memory persistently)
            try:
                for stage_cls in STAGES[2:9]:
                    chunk_context = stage_cls().run(chunk_context)
                    
                # Extract the highest-priority frame set
                frames = chunk_context.get("frames", {})
                priority = ["interpolated", "stabilized", "restored", "refined", "raw"]
                best_frames = None
                
                for key in priority:
                    if key in frames and frames[key]:
                        best_frames = frames[key]
                        break
                        
                if best_frames:
                    # Stream frames back as Base64 JPEGs
                    for frame in best_frames:
                        _, buffer = cv2.imencode('.jpg', frame)
                        frame_b64 = base64.b64encode(buffer).decode('utf-8')
                        await websocket.send_json({"frame": frame_b64})
                        
            except Exception as e:
                await websocket.send_json({"error": str(e)})
                
    except WebSocketDisconnect:
        print(f"Client {session_id} disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
    finally:
        # Cleanup session
        shutil.rmtree(f"{TMP_DIR}/{session_id}", ignore_errors=True)
