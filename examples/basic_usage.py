"""Minimal example of using RSI-Forge programmatically."""

import yaml
from rsi_forge.core.orchestrator import RSIOrchestrator
from rsi_forge.core.evaluator import Evaluator, EvalResult
from rsi_forge.core.harness import Harness


def load_config():
    with open("config/default.yaml") as f:
        return yaml.safe_load(f)


def make_task(name: str, difficulty: float = 0.5):
    def task_fn(harness: Harness, model, seed: int = 0) -> EvalResult:
        bonus = 0.05 * len(harness.extra_instructions)
        score = min(1.0, difficulty + bonus + (seed % 10) * 0.01)
        return EvalResult(task_name=name, success=score > 0.7, score=score, tokens_used=500)
    return task_fn


def main():
    cfg = load_config()
    cfg["improvement"]["max_iterations"] = 5
    tasks = {
        "coding_simple": make_task("coding_simple", 0.55),
        "math_basic": make_task("math_basic", 0.50),
        "coding_hard": make_task("coding_hard", 0.35),
        "math_contest": make_task("math_contest", 0.30),
        "agent_workspace": make_task("agent_workspace", 0.40),
    }
    evaluator = Evaluator(tasks=tasks, held_out_names=["coding_hard", "math_contest", "agent_workspace"], num_runs=2)
    orch = RSIOrchestrator(cfg)
    orch.run(evaluator)


if __name__ == "__main__":
    main()
