"""Optional W&B backend."""

from __future__ import annotations

import time
import warnings
from typing import Any


class WandbBackend:
    """Optional W&B backend. v0 logs metadata-only artifact refs."""

    def __init__(self, project: str | None = None) -> None:
        try:
            import wandb
        except ImportError as e:
            raise ImportError("Install with: pip install mil[wandb]") from e

        self._wandb = wandb
        self._run = wandb.run or wandb.init(project=project)

    def log(self, record: dict) -> None:
        self._wandb.log({"mil_record": record, "mil_when": time.time()})

    def log_artifact(self, obj: Any, name: str, content_hash: str) -> str:
        warnings.warn(
            "WandbBackend v0 logs artifact metadata only; object payload storage/retrieval "
            "is not implemented.",
            RuntimeWarning,
            stacklevel=2,
        )
        artifact = self._wandb.Artifact(name=name, type="mil-artifact")
        artifact.metadata.update(
            {"content_hash": content_hash, "python_type": type(obj).__qualname__}
        )
        self._run.log_artifact(artifact)
        ref = f"wandb:{self._run.path}/{name}:{content_hash[:12]}"
        self.log(
            {
                "kind": "tracked",
                "name": name,
                "ref": ref,
                "content_hash": content_hash,
                "type": type(obj).__qualname__,
                "when": time.time(),
            }
        )
        return ref

    def get_artifact(self, ref: str) -> Any:
        raise NotImplementedError(
            "W&B artifact retrieval is not implemented in v0; W&B refs are metadata-only."
        )

    def recent_runs(self, n: int) -> list[dict]:
        return []
