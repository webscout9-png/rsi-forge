"""Agent Harness — the evolvable system around the frozen model."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any] = Field(default_factory=dict)


class Harness(BaseModel):
    """The complete evolvable agent harness. Primary object that gets self-improved."""

    system_prompt: str = (
        "You are a capable AI assistant. Think step by step. "
        "Use tools when helpful. Be precise and honest."
    )
    tools: List[ToolSpec] = Field(default_factory=list)
    max_turns: int = 15
    context_management: str = "sliding_window"
    temperature: float = 0.7
    extra_instructions: List[str] = Field(default_factory=list)
    version: str = "0.0.0"
    parent_version: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

    def to_messages(self, user_query: str, history: Optional[List[Dict]] = None) -> List[Dict]:
        messages = [{"role": "system", "content": self._build_system_content()}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_query})
        return messages

    def _build_system_content(self) -> str:
        parts = [self.system_prompt]
        if self.extra_instructions:
            parts.append("\nAdditional instructions:\n" + "\n".join(f"- {i}" for i in self.extra_instructions))
        if self.tools:
            tool_desc = "\nAvailable tools:\n"
            for t in self.tools:
                tool_desc += f"- {t.name}: {t.description}\n"
            parts.append(tool_desc)
        return "\n\n".join(parts)

    def clone(self) -> "Harness":
        return deepcopy(self)

    def apply_patch(self, patch: Dict[str, Any]) -> "Harness":
        new = self.clone()
        if "system_prompt" in patch:
            new.system_prompt = patch["system_prompt"]
        if "add_instruction" in patch:
            new.extra_instructions.append(patch["add_instruction"])
        if "remove_instruction" in patch:
            instr = patch["remove_instruction"]
            new.extra_instructions = [i for i in new.extra_instructions if i != instr]
        if "add_tool" in patch:
            new.tools.append(ToolSpec(**patch["add_tool"]))
        if "remove_tool" in patch:
            name = patch["remove_tool"]
            new.tools = [t for t in new.tools if t.name != name]
        if "max_turns" in patch:
            new.max_turns = int(patch["max_turns"])
        if "temperature" in patch:
            new.temperature = float(patch["temperature"])
        if "context_management" in patch:
            new.context_management = patch["context_management"]
        major, minor, patch_n = map(int, new.version.split("."))
        new.version = f"{major}.{minor}.{patch_n + 1}"
        new.parent_version = self.version
        return new

    def save(self, path: str | Path):
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Harness":
        data = json.loads(Path(path).read_text())
        return cls(**data)

    def diff(self, other: "Harness") -> Dict[str, Any]:
        diffs = {}
        if self.system_prompt != other.system_prompt:
            diffs["system_prompt"] = {"from": self.system_prompt[:100] + "...", "to": other.system_prompt[:100] + "..."}
        if self.extra_instructions != other.extra_instructions:
            diffs["extra_instructions"] = {
                "added": list(set(other.extra_instructions) - set(self.extra_instructions)),
                "removed": list(set(self.extra_instructions) - set(other.extra_instructions)),
            }
        if self.max_turns != other.max_turns:
            diffs["max_turns"] = {"from": self.max_turns, "to": other.max_turns}
        return diffs
