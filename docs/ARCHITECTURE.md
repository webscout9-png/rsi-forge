# RSI-Forge Architecture

## Design Goals

1. Practical — runs on a single GPU or CPU with local models
2. Safe — held-out evaluation + regularization
3. Modular — swap models, tasks, miners, proposers
4. Research-aligned — DGM, RSIAgent, Self-Harness, RRSI, ACE
5. Evolvable

## Core Loop

Current Harness → Evaluate (train + held-out) → Collect Traces → Weakness Miner → Proposer → Validate & Gate → Accept/Reject → Update Skillbook + Archive

## Safety Mechanisms

- Held-out split the improver never sees
- Max change ratio
- Require held-out gain
- Archive of variants
- Skillbook quality gates
- Versioned harnesses with parent tracking
