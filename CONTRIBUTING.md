# Contributing

Thanks for contributing to AgentRef. Keep changes focused, tested, and easy to
review.

## Commit Convention

Use Conventional Commits:

```text
<type>(optional-scope): <summary>
```

Common types:

- `feat`: user-facing functionality
- `fix`: bug fixes
- `docs`: documentation-only changes
- `test`: tests without production code changes
- `refactor`: code changes that do not change behavior
- `perf`: performance improvements
- `chore`: build, tooling, release, or repository maintenance

Guidelines:

- Use the imperative mood: `feat: add postgres cas`.
- Keep the summary under 72 characters when practical.
- Use scopes when they clarify ownership, for example
  `feat(storage): add postgres cas`.
- Mark breaking changes with `!` and explain them in the body:
  `feat(api)!: rename runtime option`.
- Keep unrelated changes in separate commits.

## Pull Request Convention

PR titles should also follow Conventional Commits. Examples:

- `feat(storage): add postgres cas`
- `docs: add backend usage guide`
- `fix(langgraph): preserve externalized refs in node updates`

Every PR should include:

- a concise summary of what changed
- why the change is needed
- tests or checks that were run
- user-facing impact or migration notes when relevant

Prefer draft PRs for work that still needs design feedback or follow-up changes.
Request review once the PR has a clear scope, passing checks, and updated docs.

## Issue Convention

Use the issue templates for bugs and feature requests. Include enough context to
reproduce or evaluate the issue:

- expected behavior
- actual behavior
- minimal reproduction or workflow description
- relevant framework, backend, and AgentRef versions
- logs or stack traces when available

Security-sensitive reports should not be filed as public issues.
