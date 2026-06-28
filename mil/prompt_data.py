"""Prompt-pair loading helpers for demos and the explorer app."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


SELF_GROUND_ROOT = Path.home() / "p/g/n/learning/ml_research/self-ground"
DEFAULT_PROMPT_CANDIDATES = [
    SELF_GROUND_ROOT / "data/phase3_task_bank/pythia70m_negation_candidate_tasks.jsonl",
    SELF_GROUND_ROOT
    / "runs/e003_task_bank_calibration_pythia70m_margin0p1_min10/calibrated_behavioral_tasks.jsonl",
]


@dataclass
class PromptTask:
    id: str
    source_prompt: str
    target_prompt: str
    control_prompt: str
    target_tokens: list[str]
    foil_tokens: list[str]
    control_target_tokens: list[str]
    control_foil_tokens: list[str]
    metadata: dict


@dataclass
class PromptBatch:
    tasks: list[PromptTask]
    source_prompts: list[str]
    target_prompts: list[str]
    control_prompts: list[str]
    target_tokens: list[str]
    foil_tokens: list[str]
    control_target_tokens: list[str]
    control_foil_tokens: list[str]


BUILTIN_TASKS = [
    PromptTask(
        id="builtin-movie-negative",
        source_prompt="The movie was not positive. The movie was",
        target_prompt="The movie was not positive. The movie was",
        control_prompt="The movie was positive. The movie was",
        target_tokens=[" negative"],
        foil_tokens=[" positive"],
        control_target_tokens=[" positive"],
        control_foil_tokens=[" negative"],
        metadata={"source": "builtin", "family": "sentiment_negation"},
    ),
    PromptTask(
        id="builtin-service-unhelpful",
        source_prompt="The service was not helpful. The service was",
        target_prompt="The service was not helpful. The service was",
        control_prompt="The service was helpful. The service was",
        target_tokens=[" unhelpful"],
        foil_tokens=[" helpful"],
        control_target_tokens=[" helpful"],
        control_foil_tokens=[" unhelpful"],
        metadata={"source": "builtin", "family": "sentiment_negation"},
    ),
]


def find_default_prompts_file() -> Path | None:
    for path in DEFAULT_PROMPT_CANDIDATES:
        if path.exists():
            return path
    return None


def load_prompt_tasks(path: str | Path | None = None, *, limit: int | None = None) -> list[PromptTask]:
    if path is None:
        found = find_default_prompts_file()
        if found is None:
            raise FileNotFoundError(
                "Could not locate Paul's SELF-GROUND prompt task file. "
                "Pass --prompts-file with a JSONL task-bank path."
            )
        path = found
    rows = _read_jsonl(Path(path))
    tasks = [_row_to_task(row) for row in rows]
    if limit is not None:
        tasks = tasks[:limit]
    if not tasks:
        raise ValueError(f"No prompt tasks loaded from {path}")
    return tasks


def builtin_tasks(limit: int | None = None) -> list[PromptTask]:
    return BUILTIN_TASKS[:limit]


def make_prompt_batch(tasks: Iterable[PromptTask]) -> PromptBatch:
    task_list = list(tasks)
    return PromptBatch(
        tasks=task_list,
        source_prompts=[task.source_prompt for task in task_list],
        target_prompts=[task.target_prompt for task in task_list],
        control_prompts=[task.control_prompt for task in task_list],
        target_tokens=[_one(task.target_tokens, task.id, "target_tokens") for task in task_list],
        foil_tokens=[_one(task.foil_tokens, task.id, "foil_tokens") for task in task_list],
        control_target_tokens=[
            _one(task.control_target_tokens, task.id, "control_target_tokens") for task in task_list
        ],
        control_foil_tokens=[
            _one(task.control_foil_tokens, task.id, "control_foil_tokens") for task in task_list
        ],
    )


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _row_to_task(row: dict) -> PromptTask:
    required = {
        "prompt",
        "control_prompt",
        "target_tokens",
        "foil_tokens",
        "control_target_tokens",
        "control_foil_tokens",
    }
    missing = sorted(required - row.keys())
    if missing:
        raise ValueError(
            "Prompt rows must use the SELF-GROUND phase3 task-bank schema. "
            f"Missing keys for row {row.get('id', '<unknown>')}: {missing}"
        )
    return PromptTask(
        id=str(row.get("id", "")),
        source_prompt=str(row.get("source_prompt", row["prompt"])),
        target_prompt=str(row["prompt"]),
        control_prompt=str(row["control_prompt"]),
        target_tokens=[str(x) for x in row["target_tokens"]],
        foil_tokens=[str(x) for x in row["foil_tokens"]],
        control_target_tokens=[str(x) for x in row["control_target_tokens"]],
        control_foil_tokens=[str(x) for x in row["control_foil_tokens"]],
        metadata=dict(row.get("metadata", {})),
    )


def _one(values: list[str], task_id: str, field: str) -> str:
    if len(values) != 1:
        raise ValueError(f"{field} for {task_id} must contain exactly one token.")
    return values[0]
