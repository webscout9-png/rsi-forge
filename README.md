# RSI-Forge 🔥

**Recursive Self-Improvement Forge for Open-Source AI Models**

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

> Drop in any open-source model → it improves its own harness, memory, and tools → becomes measurably smarter over time.

RSI-Forge is a practical, research-backed framework for **bounded recursive self-improvement** of open-source language models. It focuses on what actually works today: evolving the agent harness and persistent skill memory under strict evaluation gates, with an optional path to weight updates later.

Built by synthesizing the strongest ideas from 2025–2026 research:
- Darwin Gödel Machine (open-ended archive + code evolution)
- RSIAgent (broad-then-deep exploration + reusable memory)
- Self-Harness / RRSI (regularized harness improvement)
- Agentic Context Engine / recursive-improve (trace → skillbook)
- Self-Rewarding Language Models (Meta)

---

## Why RSI-Forge?

Most self-improvement demos either overfit, require enormous compute, or lack proper held-out evaluation. RSI-Forge is designed differently:

| Principle | How we implement it |
|-----------|---------------------|
| **Held-out is sacred** | Improver never sees the real test distribution |
| **Regularization first** | Max change ratio + require held-out gain |
| **Harness before weights** | Highest ROI, lowest risk |
| **Archive everything** | Darwinian open-ended exploration of variants |
| **Modular & swappable** | Models, miners, proposers, evaluators are independent |

## Features

- Plug-and-play with Hugging Face, vLLM, Ollama, or any OpenAI-compatible endpoint
- Automatic harness evolution (system prompt, tools, control flow, instructions)
- Persistent Skillbook with quality gates
- Weakness mining from execution traces
- Regularized proposal & selection (anti-overfitting)
- Full version control + automatic rollback support
- Clean CLI + YAML configuration
- Ready for overnight autonomous runs

## Quick Start

```bash
# Clone
git clone https://github.com/webscout9-png/rsi-forge.git
cd rsi-forge

# Create environment
python -m venv .venv
source .venv/bin/activate

# Install
pip install -e .

# Initialize project
rsi-forge init

# Edit config/default.yaml — set your model & backend

# Run a self-improvement cycle
rsi-forge improve --iterations 5
```

### Programmatic usage

```python
from rsi_forge import RSIOrchestrator, Harness
import yaml

with open("config/default.yaml") as f:
    config = yaml.safe_load(f)

orch = RSIOrchestrator(config)
# Attach your real Evaluator and call orch.run(evaluator)
```

## Project Structure

```
rsi-forge/
├── config/default.yaml          # Main configuration
├── src/rsi_forge/
│   ├── core/                    # Harness, Model, Evaluator, Orchestrator
│   ├── improver/                # WeaknessMiner + Proposer
│   ├── memory/                  # Skillbook
│   ├── fine_tune/               # (Future) LoRA / self-reward loops
│   └── cli.py                   # Typer CLI
├── examples/
├── tests/
├── docs/ARCHITECTURE.md
└── pyproject.toml
```

## Configuration Highlights

```yaml
model:
  name: "Qwen/Qwen2.5-7B-Instruct"
  backend: "transformers"   # or vllm / ollama / openai

improvement:
  levels: ["harness", "skillbook"]
  regularization:
    enabled: true
    require_held_out_gain: true

evaluation:
  held_out_tasks: [...]     # Never shown to the improver
```

See `config/default.yaml` for the full set of options.

## Safety Notes

Recursive self-improvement is powerful. RSI-Forge is deliberately **bounded**:

- Start with small models and verifiable tasks (code, math)
- Always keep a large held-out set
- Enable regularization
- Review major patches before long autonomous runs
- Monitor for reward hacking and collapse

This is a research and engineering tool, not an unbounded self-improving AGI.

## Roadmap

- [x] Core harness + skillbook RSI loop
- [x] Regularized proposal & held-out gating
- [x] Multi-backend model support
- [ ] Real benchmark integrations (HumanEval, MATH, agent suites)
- [ ] Full agent trace instrumentation
- [ ] Optional LoRA self-rewarding loop
- [ ] Multi-agent (Curriculum / Actor / Verifier) variant
- [ ] Meta-improvement of the miner/proposer themselves

## Citation

If you use RSI-Forge in research, please cite the underlying works that inspired it (Darwin Gödel Machine, RSIAgent, Self-Harness, RRSI, Self-Rewarding LMs, ACE) and this repository.

## License

Apache 2.0 — free for research and commercial use. See [LICENSE](LICENSE).

## Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md).

---

**Built for the open-source community.**  
Let’s make models that genuinely get smarter — safely and measurably.
