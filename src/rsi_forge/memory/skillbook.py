"""Persistent Skillbook — evolving memory of useful strategies."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


class Skill(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    description: str
    strategy: str
    evidence: List[str] = Field(default_factory=list)
    success_count: int = 0
    failure_count: int = 0
    quality_score: float = 0.5
    tags: List[str] = Field(default_factory=list)
    created_at: float = Field(default_factory=time.time)
    last_used: float = Field(default_factory=time.time)
    source: str = "mined"


class Skillbook(BaseModel):
    skills: List[Skill] = Field(default_factory=list)
    max_skills: int = 200
    quality_threshold: float = 0.7
    version: int = 1

    def add_skill(self, skill: Skill) -> bool:
        if skill.quality_score < self.quality_threshold:
            return False
        for existing in self.skills:
            if existing.title.lower() == skill.title.lower():
                existing.evidence.extend(skill.evidence)
                existing.success_count += skill.success_count
                existing.quality_score = max(existing.quality_score, skill.quality_score)
                return True
        self.skills.append(skill)
        self._prune()
        return True

    def _prune(self):
        if len(self.skills) <= self.max_skills:
            return
        self.skills.sort(key=lambda s: (s.quality_score, s.success_count), reverse=True)
        self.skills = self.skills[: self.max_skills]

    def get_relevant(self, query: str, k: int = 5) -> List[Skill]:
        query_lower = query.lower()
        scored = []
        for skill in self.skills:
            score = skill.quality_score
            if any(tag.lower() in query_lower for tag in skill.tags):
                score += 2
            if any(w in skill.title.lower() for w in query_lower.split()):
                score += 1
            scored.append((score, skill))
        scored.sort(reverse=True)
        return [s for _, s in scored[:k]]

    def to_prompt_section(self, query: str, k: int = 5) -> str:
        relevant = self.get_relevant(query, k=k)
        if not relevant:
            return ""
        lines = ["### Relevant Skills & Strategies"]
        for i, skill in enumerate(relevant, 1):
            lines.append(f"{i}. **{skill.title}**: {skill.strategy}")
        return "\n".join(lines)

    def save(self, path: str | Path):
        Path(path).write_text(self.model_dump_json(indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "Skillbook":
        p = Path(path)
        if not p.exists():
            return cls()
        return cls(**json.loads(p.read_text()))

    def summary(self) -> str:
        n = len(self.skills)
        avg = sum(s.quality_score for s in self.skills) / max(1, n)
        return f"Skillbook v{self.version}: {n} skills, avg quality {avg:.2f}"
