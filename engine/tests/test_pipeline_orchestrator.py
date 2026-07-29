import pytest
from app.pipeline.orchestrator import STAGES, _STAGE_MAP, run_pipeline

def test_stages_order_matches_stage_map():
    """
    Issue #1: Ensures that the order of stages in STAGES list matches
    the execution order implicitly defined by _STAGE_MAP IDs.
    """
    # Create an inverse map from stage_name to its _STAGE_MAP ID
    inverse_map = {name: id for id, name in _STAGE_MAP.items()}
    
    last_id = -1
    for stage in STAGES:
        # Some stages might not be in the _STAGE_MAP if they don't have a specific skip ID
        if stage.name in inverse_map:
            current_id = inverse_map[stage.name]
            assert current_id > last_id, f"Stage {stage.name} is out of order! Found map ID {current_id} after {last_id}"
            last_id = current_id

def test_invalid_skip_to_stage_raises_value_error():
    """
    Issue #5: Invalid skip_to_stage fails silently.
    """
    with pytest.raises(ValueError, match="Invalid skip_to_stage: 999"):
        run_pipeline(
            job_id="test_invalid_skip",
            image="dummy.jpg",
            audio="dummy.wav",
            skip_to_stage=999,
            cached_job_id="dummy_job_id"
        )

def test_identity_name_with_invalid_skip_raises_value_error(monkeypatch):
    """
    Issue #2: identity_name + skip_to_stage combination can crash or clobber state.
    """
    # Mock load_identity to avoid hitting DB/disk
    def mock_load_identity(name):
        import numpy as np
        return np.zeros(512), "dummy_thumbnail.jpg"
    
    monkeypatch.setattr("app.pipeline.orchestrator.load_identity", mock_load_identity)
    monkeypatch.setattr("cv2.imread", lambda path: "dummy_image_data")
    monkeypatch.setattr("app.pipeline.orchestrator.validate_all_checkpoints", lambda *args, **kwargs: None)
    monkeypatch.setattr("os.makedirs", lambda *args, **kwargs: None)
    
    # Missing image, identity_name provided, but skipping to Stage 1
    with pytest.raises(ValueError, match="Cannot skip to stage 1 or 2 with an identity_name but no input image"):
        run_pipeline(
            job_id="test_identity_skip",
            image=None,
            identity_name="test_identity",
            skip_to_stage=1,
            cached_job_id="dummy_job_id"
        )
