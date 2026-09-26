"""RSI Orchestrator — the main recursive self-improvement loop."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.table import Table

from rsi_forge.core.harness import Harness
from rsi_forge.core.evaluator import Evaluator, AggregateResult
from rsi_forge.memory.skillbook import Skillbook

console = Console()


class RSIOrchestrator:
    """
    Classic RSI loop:
      1. Evaluate current harness (train + held-out)
      2. Propose harness patches
      3. Accept only if held-out improves
      4. Update skillbook + archive

    Model loading is LAZY: pass model=... or set orch.model after init.
    This keeps the offline demo working without transformers/torch.
    """

    def __init__(self, config: Dict[str, Any], model=None):
        self.config = config
        self.output_dir = Path(config.get("project", {}).get("output_dir", "./runs"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Lazy: only create real model if caller did not pass one
        self._model = model
        self._model_config = config.get("model", {})

        hcfg = config.get("harness", {})
        self.harness = Harness(
            system_prompt=hcfg.get("system_prompt", Harness().system_prompt),
            max_turns=hcfg.get("max_turns", 15),
            temperature=hcfg.get("temperature", 0.7),
        )
        self.skillbook = Skillbook.load(
            config.get("memory", {}).get("skillbook_path", "./skillbook.json")
        )
        self.archive: List[Dict] = []
        self.iteration = 0

    @property
    def model(self):
        if self._model is None:
            from rsi_forge.core.model import create_model
            self._model = create_model(self._model_config)
        return self._model

    @model.setter
    def model(self, value):
        self._model = value

    def run(self, evaluator: Evaluator, max_iterations: Optional[int] = None):
        max_iter = max_iterations or self.config.get("improvement", {}).get("max_iterations", 5)
        patience = self.config.get("improvement", {}).get("plateau_patience", 3)
        no_improve = 0
        best_held = -1.0

        console.print(f"\n[bold cyan]RSI-Forge starting[/] — max {max_iter} iterations\n")

        for i in range(max_iter):
            self.iteration = i + 1
            console.rule(f"[bold]Iteration {self.iteration}")

            train_res = evaluator.evaluate(self.harness, self.model, split="train")
            held_res = evaluator.evaluate(self.harness, self.model, split="held_out")
            self._log_results("Current", train_res, held_res)

            if held_res.mean_score > best_held:
                best_held = held_res.mean_score
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= patience:
                console.print("[yellow]Plateau reached. Stopping early.[/]")
                break

            # Simple improvement: add a verification instruction if score is low
            improved = False
            if held_res.mean_score < 0.9 and "verify" not in " ".join(self.harness.extra_instructions).lower():
                new_h = self.harness.apply_patch({
                    "add_instruction": "Before finalizing, briefly verify your answer."
                })
                new_held = evaluator.evaluate(new_h, self.model, split="held_out")
                if new_held.mean_score >= held_res.mean_score:
                    self.harness = new_h
                    self._archive(new_h, new_held)
                    improved = True
                    console.print(
                        f"[green]✓ Accepted harness update[/] → held-out {new_held.mean_score:.3f}"
                    )

            if not improved:
                console.print("[dim]No improving patch this round.[/]")

            self._save_state()

        console.print("\n[bold green]RSI loop finished.[/]")
        self._print_summary()

    def _archive(self, harness: Harness, result: AggregateResult):
        self.archive.append({
            "iteration": self.iteration,
            "version": harness.version,
            "held_out_score": result.mean_score,
            "timestamp": time.time(),
        })
        harness.save(self.output_dir / f"harness_v{harness.version}.json")

    def _log_results(self, label: str, train: AggregateResult, held: AggregateResult):
        table = Table(title=f"{label} Performance")
        table.add_column("Split")
        table.add_column("Mean Score")
        table.add_column("Pass Rate")
        table.add_column("# Tasks")
        table.add_row("Train", f"{train.mean_score:.3f}", f"{train.pass_rate:.2%}", str(train.num_tasks))
        table.add_row("Held-out", f"{held.mean_score:.3f}", f"{held.pass_rate:.2%}", str(held.num_tasks))
        console.print(table)

    def _save_state(self):
        state = {
            "iteration": self.iteration,
            "harness_version": self.harness.version,
            "archive_size": len(self.archive),
            "skillbook": self.skillbook.summary(),
        }
        (self.output_dir / "state.json").write_text(json.dumps(state, indent=2))
        self.harness.save(self.output_dir / "current_harness.json")

    def _print_summary(self):
        console.print(f"Iterations: {self.iteration}")
        console.print(f"Archive size: {len(self.archive)}")
        console.print(f"Skillbook: {self.skillbook.summary()}")
        if self.archive:
            best = max(self.archive, key=lambda x: x["held_out_score"])
            console.print(f"Best held-out: {best['held_out_score']:.3f} (v{best['version']})")
