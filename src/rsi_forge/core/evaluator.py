"""Evaluation suite with held-out protection and regularization metrics."""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from rsi_forge.core.harness import Harness


@dataclass
class EvalResult:
    task_name: str
    success: bool
    score: float
    tokens_used: int = 0
    latency: float = 0.0
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AggregateResult:
    mean_score: float
    pass_rate: float
    token_efficiency: float
    num_tasks: int
    per_task: Dict[str, float]
    is_held_out: bool = False


class Evaluator:
    """Runs a harness against tasks with train vs held-out splits."""

    def __init__(
        self,
        tasks: Dict[str, Callable],
        held_out_names: List[str] = None,
        num_runs: int = 3,
        seed: int = 42,
    ):
        self.tasks = tasks
        self.held_out_names = set(held_out_names or [])
        self.num_runs = num_runs
        self.seed = seed

    def evaluate(
        self,
        harness: Harness,
        model,
        split: str = "train",
        max_tasks: Optional[int] = None,
    ) -> AggregateResult:
        if split == "train":
            task_names = [n for n in self.tasks if n not in self.held_out_names]
        elif split == "held_out":
            task_names = [n for n in self.tasks if n in self.held_out_names]
        else:
            task_names = list(self.tasks.keys())

        if max_tasks:
            task_names = task_names[:max_tasks]

        all_scores = []
        per_task = {}
        total_tokens = 0
        rng = random.Random(self.seed)

        for name in task_names:
            task_fn = self.tasks[name]
            scores = []
            for _ in range(self.num_runs):
                result = task_fn(harness, model, seed=rng.randint(0, 10_000))
                scores.append(result.score)
                total_tokens += result.tokens_used
            mean = sum(scores) / len(scores) if scores else 0.0
            per_task[name] = mean
            all_scores.append(mean)

        if not all_scores:
            return AggregateResult(0.0, 0.0, 0.0, 0, {}, is_held_out=(split == "held_out"))

        mean_score = sum(all_scores) / len(all_scores)
        pass_rate = sum(1 for s in all_scores if s >= 0.99) / len(all_scores)
        token_eff = mean_score / max(total_tokens / max(len(all_scores), 1), 1e-6)

        return AggregateResult(
            mean_score=mean_score,
            pass_rate=pass_rate,
            token_efficiency=token_eff,
            num_tasks=len(all_scores),
            per_task=per_task,
            is_held_out=(split == "held_out"),
        )
