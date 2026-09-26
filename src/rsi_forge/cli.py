"""Command-line interface for RSI-Forge."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer
import yaml
from rich.console import Console

from rsi_forge.core.orchestrator import RSIOrchestrator
from rsi_forge.core.evaluator import Evaluator, EvalResult
from rsi_forge.core.harness import Harness

app = typer.Typer(name="rsi-forge", help="Recursive Self-Improvement Forge")
console = Console()


def load_config(path: str = "config/default.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


@app.command()
def init(
    output_dir: str = typer.Option("./runs", help="Where to store run artifacts"),
):
    """Initialize a new RSI-Forge project."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path("skillbook.json").touch()
    console.print("[green]✓ RSI-Forge project initialized[/]")
    console.print("Edit config/default.yaml then run: rsi-forge improve")


@app.command()
def improve(
    config: str = typer.Option("config/default.yaml", help="Path to config YAML"),
    iterations: int = typer.Option(None, help="Override max iterations"),
    model: Optional[str] = typer.Option(None, help="Override model name"),
):
    """Run the recursive self-improvement loop (demo with dummy tasks)."""
    cfg = load_config(config)
    if model:
        cfg["model"]["name"] = model
    if iterations:
        cfg["improvement"]["max_iterations"] = iterations

    def make_dummy_task(name: str, base_score: float = 0.5):
        def task(harness: Harness, model, seed: int = 0) -> EvalResult:
            # Simulate that better harnesses score higher
            bonus = 0.05 * len(harness.extra_instructions)
            score = min(1.0, base_score + bonus + (seed % 10) * 0.01)
            return EvalResult(task_name=name, success=score > 0.7, score=score, tokens_used=100)
        return task

    tasks = {
        "coding_simple": make_dummy_task("coding_simple", 0.55),
        "math_basic": make_dummy_task("math_basic", 0.50),
        "coding_hard": make_dummy_task("coding_hard", 0.35),
        "math_contest": make_dummy_task("math_contest", 0.30),
        "agent_workspace": make_dummy_task("agent_workspace", 0.40),
    }

    evaluator = Evaluator(
        tasks=tasks,
        held_out_names=["coding_hard", "math_contest", "agent_workspace"],
        num_runs=2,
    )

    # Use a lightweight fake model so the demo runs without GPU/API
    class FakeModel:
        def generate(self, messages, config):
            return "ok"

    orch = RSIOrchestrator(cfg)
    orch.model = FakeModel()  # demo mode — no real LLM needed
    orch.run(evaluator, max_iterations=iterations or 3)


@app.command()
def status(
    run_dir: str = typer.Option("./runs", help="Run directory"),
):
    """Show status of the current or last run."""
    state_file = Path(run_dir) / "state.json"
    if not state_file.exists():
        console.print("[yellow]No run found. Run: rsi-forge improve[/]")
        return
    state = json.loads(state_file.read_text())
    console.print_json(data=state)


@app.command()
def show_harness(
    path: str = typer.Option("./runs/current_harness.json", help="Harness JSON path"),
):
    """Display the current harness."""
    if not Path(path).exists():
        console.print(f"[yellow]No harness at {path}. Run improve first.[/]")
        return
    h = Harness.load(path)
    console.print(h.model_dump_json(indent=2))


if __name__ == "__main__":
    app()
