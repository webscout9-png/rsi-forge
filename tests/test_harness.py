"""Basic tests for the Harness and Skillbook."""

import pytest
from rsi_forge.core.harness import Harness
from rsi_forge.memory.skillbook import Skillbook, Skill


def test_harness_creation():
    h = Harness()
    assert h.version == "0.0.0"
    assert len(h.system_prompt) > 10


def test_harness_apply_patch():
    h = Harness()
    new = h.apply_patch({"add_instruction": "Always verify your answer."})
    assert "Always verify your answer." in new.extra_instructions
    assert new.version != h.version
    assert new.parent_version == h.version


def test_skillbook_add_and_retrieve():
    sb = Skillbook(max_skills=10, quality_threshold=0.5)
    skill = Skill(
        title="Plan first",
        description="Write a plan before acting",
        strategy="Always outline steps before using tools.",
        quality_score=0.9,
        tags=["planning"],
    )
    assert sb.add_skill(skill) is True
    assert len(sb.skills) == 1
