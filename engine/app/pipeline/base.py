"""
Common interface every pipeline stage follows. Keeping this tiny and
consistent is what lets you swap a stage's model without touching the
orchestrator or any other stage.
"""
from abc import ABC, abstractmethod
from typing import Any


class PipelineStage(ABC):
    name: str = "unnamed_stage"

    @abstractmethod
    def run(self, context: dict) -> dict:
        """
        Read whatever this stage needs from `context`, do its work,
        write results back into `context` under this stage's own key(s),
        and return the (mutated) context for the next stage.
        """
        raise NotImplementedError
