1. Writing code: simplicity first
Goal: the minimum code that solves the stated problem. Nothing speculative.

No features beyond what was asked.
No abstractions for single-use code. No configurability, flexibility, or hooks that were not requested.
No error handling for impossible scenarios. Handle the failures that can actually happen.
If the solution runs 200 lines and could be 50, rewrite it before showing it.
If you find yourself adding "for future extensibility", stop. Future extensibility is a future decision.
Bias toward deleting code over adding code. Shipping less is almost always better.
The test: would a senior engineer reading the diff call this overcomplicated? If yes, simplify.

2. Surgical changes
Goal: clean, reviewable diffs. Change only what the request requires.

Do not "improve" adjacent code, comments, formatting, or imports that are not part of the task.
Do not refactor code that works just because you are in the file.
Do not delete pre-existing dead code unless asked. If you notice it, mention it in the summary.
Do clean up orphans created by your own changes (unused imports, variables, functions your edit made obsolete).
Match the project's existing style exactly: indentation, quotes, naming, file layout.
The test: every changed line traces directly to the user's request. If a line fails that test, revert it.