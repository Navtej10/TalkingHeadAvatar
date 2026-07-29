"""
Common interface every pipeline stage follows. Keeping this tiny and
consistent is what lets you swap a stage's model without touching the
orchestrator or any other stage.
"""
from abc import ABC, abstractmethod
from typing import Any


class PipelineStageError(Exception):
    def __init__(self, stage_name, job_id, original_exception):
        super().__init__(f"Stage {stage_name} failed for job {job_id}: {original_exception}")
        self.stage_name = stage_name
        self.job_id = job_id
        self.original_exception = original_exception

class PipelineStage(ABC):
    """
    Base class for all pipeline stages.
    
    CONTRACT: PipelineStage instances are shared singletons across concurrent jobs.
    Do NOT store per-job data or mutable state on `self` in `__init__` or `run()`.
    All per-job data must be read from and written to the `context` dictionary.
    """
    name: str = "unnamed_stage"

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Read whatever this stage needs from `context`, do its work,
        write results back into `context` under this stage's own key(s),
        and return the (mutated) context for the next stage.
        """
        raise NotImplementedError
