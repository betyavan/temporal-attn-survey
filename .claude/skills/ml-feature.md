---
name: ml-feature
description: Implement an ML feature in the temporal-attn-survey repo (CogVideoX hooks, ablation, metrics) by orchestrating ml-coder and ml-reviewer. Runs a plan-audit in parallel with the first implementation pass when the task is large (e.g. "implement Phase 1", "implement src/hooks.py"), then runs ml-reviewer on the diff and loops back to ml-coder if CRITICAL/MAJOR findings appear, up to 3 iterations. Invoke this skill whenever the user asks to write or modify code in `src/hooks.py`, `src/ablation.py`, `src/metrics.py`, or any new ML module in this repo.
---

You are orchestrating two subagents — `ml-coder` (writes ML code) and `ml-reviewer` (read-only review) — to implement an ML feature in `/devstorage/temporal-attn-survey`.

## Inputs you need

Before launching anything, make sure you have:
- **The feature spec.** Usually a section of `README.md` (e.g., "Phase 1, Step 2 — Extracting attention via hooks"). If the user's request is vague, ask one clarifying question before proceeding.
- **The target files.** Confirm which of `src/hooks.py`, `src/ablation.py`, `src/metrics.py` are in scope. If a new module is needed, ask before creating it (the canonical structure is three files; deviations need justification).
- **Task size.** A "small" task touches one function; a "large" task is a whole module or a whole README phase.

## Workflow

### Step 1 — Plan-audit + implementation (parallel for large tasks)

For **large** tasks, send a single message with two Agent calls in parallel:

1. **Plan-audit agent** (`subagent_type: general-purpose`): read `README.md` and any existing code, then report — in under 200 words — whether the plan in the user's request matches the canonical spec, what's ambiguous, and what the implementer should be careful about. This is a sanity check that runs in parallel and does not block.
2. **ml-coder**: implement the feature per the spec.

For **small** tasks (single function, obvious change), skip the plan-audit and call `ml-coder` directly.

Pass ml-coder a self-contained prompt: feature description, the exact README section(s) to follow, the target file path(s), and any constraints from the conversation. Ml-coder will return a report of files changed and lint/test status.

### Step 2 — Review

Once ml-coder returns, invoke `ml-reviewer` (foreground). Hand it:
- The list of files ml-coder changed.
- A one-line summary of what was implemented.
- Any spec deviations ml-coder flagged.

Ml-reviewer reads the diff itself and returns a structured report (CRITICAL / MAJOR / MINOR + lint/test results + spec compliance).

### Step 3 — Fix loop (up to 3 iterations)

- If the review has **no CRITICAL or MAJOR findings**: report success to the user with a terse summary (files changed, key decisions, link to review report). Done.
- If there are CRITICAL or MAJOR findings: re-invoke `ml-coder` with the review report attached and a directive: *"Address the CRITICAL and MAJOR findings below. MINOR findings are optional. Do not introduce new functionality."* Then go back to Step 2.
- **Hard limit: 3 iterations.** If the third review still has CRITICAL findings, stop the loop and hand control back to the user with:
  - All three review reports in summary form.
  - Your read on what the recurring issue is.
  - A specific question for the user (e.g., "the head-dim derivation needs a design call — should we expose it as an argument or detect it via attn.heads?").

### Step 4 — Final report

After a clean review (or after hitting the iteration cap), write a 4–6 line summary to the user:
- Files created/modified (with absolute paths).
- Which tests / lint passed.
- Any deviations from the README spec, with justification.
- If the loop terminated unclean: what's still broken and what decision is needed.

Do not dump the full reviewer output to the user — distill it.

## Rules

- **Never edit files yourself.** Your role is orchestration. Editing happens in ml-coder; reading-only review happens in ml-reviewer.
- **Run agents in parallel only when they're independent.** Plan-audit and ml-coder are independent (audit reads, coder writes — different file roles). Ml-reviewer depends on ml-coder's output, so it always runs sequentially after.
- **Distill, don't relay.** The reviewer's report can be 50+ lines; the user wants the punchline.
- **Stay in the orchestrator role even when tempted.** If the user adds a follow-up like "actually also add X", route it through ml-coder rather than editing yourself — it keeps the review cycle honest.
- **Respect the repo conventions.** mypy --strict, no full-attention materialization, head-level ablation on `to_out`, notebooks stay thin. These are enforced by ml-coder and ml-reviewer; you don't need to repeat them, but if a user request would violate one, push back before launching agents.
