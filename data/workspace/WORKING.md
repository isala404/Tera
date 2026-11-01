<!-- generated: tera, edits are overwritten; put yours in PERSONA.md -->
# Doing the work

For code, files, git, installs, delegation and model choice.

## Method

Read before you write, and understand why something is shaped the way it is before changing it. Define done as something checkable and verify it before claiming success. A change is finished when every place encoding the same fact agrees. Touch only what the task needs, propose first when the work is architectural or ambiguous, and stop and ask after repeated failed fixes. Write simple, boring code that matches what's already there.

## Tools

Python through `uv`, never bare `pip`. JavaScript and TypeScript through `bun`. Docker with `--rm` for services or a pinned OS. `nix shell` or a throwaway container for tools that aren't installed. Install permanently only when asked, and `SYSTEM.md` says whether this machine is declaratively managed.

## Delegation

Heavy work goes to `projects/<id>/` when ongoing or `tasks/<id>/` when single use or scheduled, never into the middle of a conversation. Each has its own `AGENTS.md` and a `MEMORY.md` that survives runs. Return the outcome, not the transcript.

## Models

Conversation, schedules and maintenance use the Codex settings in `{{WORKSPACE}}/.codex-home/config.toml`, and `$model-config` reports or changes them. Codex owns subagent delegation and model selection, so let Codex choose the worker model unless {{OWNER}} names one.

## Skill work

Create or improve a skill only after {{OWNER}} approves the specific candidate, with `$skill-creator`. Read the existing skill first and keep its name. Keep `SKILL.md` lean, move repeated mechanics into compact executable scripts, test them directly, and do not create `agents/openai.yaml`.

## Never

- Kill processes you didn't start. A bound port is something to report.
- Discard uncommitted work or stage blindly. Read `git status` and stage paths by name.
- Edit generated files, lockfiles or installed copies by hand. Change the source and regenerate.
- Overwrite a file you haven't read.
- Put a secret anywhere it can be read back, including logs, scratch files and messages.
- Run work in the foreground that outlives your turn, or `rm -rf` a variable path without proving it's set.

## Commits and PRs

Follow the repository's own log. Otherwise use a short imperative subject, a body only when the why needs it, and no AI attribution. PR descriptions lead with the problem and the fix in plain English, then what's verified and what's open.
