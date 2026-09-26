# Contributing to RSI-Forge

Thank you for your interest in making open-source models smarter through recursive self-improvement.

## Ways to Contribute

- Bug reports & fixes
- New evaluation tasks
- Better miners / proposers
- Additional model backends
- Documentation & examples
- Safety research

## Development Setup

```bash
git clone https://github.com/webscout9-png/rsi-forge.git
cd rsi-forge
python -m venv .venv
source .venv/bin/activate
pip install -e ".[all]"
```

## Testing

```bash
pytest tests/ -v
```

## Design Principles

1. Held-out evaluation is sacred
2. Regularization first
3. Modularity
4. Safety
5. Open science
