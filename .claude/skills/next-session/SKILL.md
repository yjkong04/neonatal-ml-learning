---
name: next-session
description: Surveys this repo's git branches, README, STUDY_PLAN, and math study plan to recommend ONE concrete, output-heavy task for a 1-2 hour work session. Use when starting a session on neonatal-ml-learning and you don't want to spend time deciding what to work on.
---

# Next Session

Hand back one concrete next action for a 1-2 hour block — not a status report, not a menu. Bias toward resuming in-flight work over starting fresh, and toward producing output (notebook code, project code, practice problems) over passive reading when both are viable.

## Instructions

### Step 1 — Check for in-flight work first
Run `git branch -vv` to see long-lived branches (`project/*`, `notes/*`, `math/*`) and `git log --all --oneline -20` for recent activity across them. Run `git status` on the current branch.

If there's an uncommitted change, a half-finished notebook cell, or a branch that's clearly mid-task — **resume that first.** Lowest context-switch cost, context is already loaded. Skip straight to Step 4.

### Step 2 — If nothing is in flight, read the planning docs
In this order:
1. `README.md` — the Progress table (✅ / 🚧 / ⏸️ rows)
2. `STUDY_PLAN.md` — the Status section, the active Phase's Checkpoints/Deliverables, and the Math Foundations section
3. `resources/math_study_plan.md` — if math is the live track, find the next incomplete Day (1–7)
4. Spot-check `notebooks/` and `projects/` for partially-built work that isn't yet reflected in README/STUDY_PLAN — docs can lag reality, trust the files over the table when they disagree

### Step 3 — Cross-reference and pick ONE task
Priority order:
- A 🚧 in-progress item beats a ⏸️ queued one — finish what's started before opening something new.
- Among ties, prefer whatever unblocks the most other work (e.g., Stage 1.1 probability gates the Phase 5 calibration work; the math plan's own rule is "if shaky, fix before moving on").
- Default to a deliverable-producing task over passive reading, unless the next concept is a genuine prerequisite for the next deliverable.
- Scope to 1–2 hours. If the next checkpoint is bigger, name the first sub-piece only (e.g., "get PhysioNet 2017 ECG data loading working," not "finish Phase 3").

### Step 4 — Output format
Direct and short:
- **Branch:** which branch to check out, or the exact `git checkout -b ...` for a new one
- **File/task:** the exact file or notebook to open and what to do in it
- **Why this one:** one sentence, only if non-obvious
- **Done when:** a concrete stopping point for the block

No alternatives list, no backlog dump. One task. If it's wrong, she'll redirect.

### Step 5 — Read-only
Don't edit README.md or STUDY_PLAN.md progress markers, and don't add "today" notes anywhere. `git log` and the next commit/PR are the source of truth for what got done — this skill only reads state, it doesn't record it.
