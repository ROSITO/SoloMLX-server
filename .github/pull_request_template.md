## What and why

<!-- What does this change, and what problem does it solve? Link the issue: "Fixes #123". -->

## How it was tested

<!-- Commands run and results. For inference changes: Mac (chip, RAM), model, prompt size. -->

- [ ] `.venv/bin/python -m pytest -q` passes
- [ ] Checked on a real model (which one: ...), if generation, caches, memory or tool calls are affected

## Numbers

<!-- For performance changes: before / after (tok/s, latency, peak memory), same machine and prompt. Delete if not relevant. -->

## Checklist

- [ ] Focused change (one fix or feature)
- [ ] Tests added or updated for behaviour changes
- [ ] README updated if a `MLXSERVE_*` setting, endpoint or CLI command changed
- [ ] No unrelated `bench/` result files
