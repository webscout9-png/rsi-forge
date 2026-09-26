"""RSI-Forge: Recursive Self-Improvement Forge for Open-Source AI Models."""

__version__ = "0.1.0"
__author__ = "RSI-Forge Contributors"

from rsi_forge.core.orchestrator import RSIOrchestrator
from rsi_forge.core.harness import Harness
from rsi_forge.memory.skillbook import Skillbook

__all__ = ["RSIOrchestrator", "Harness", "Skillbook", "__version__"]
