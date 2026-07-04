# Contributing to AMOS

Thanks for helping improve AMOS. This project accepts contributions under the
Apache License 2.0 and uses the Developer Certificate of Origin for contributor
certification.

## License

By contributing to this repository, you agree that your contribution is
licensed under the Apache License 2.0 unless a file explicitly states
otherwise.

See [LICENSE](LICENSE) for the license terms.

## Developer Certificate of Origin

All commits must include a DCO sign-off. The sign-off certifies that you have
the right to submit the contribution under this project's license.

Use `git commit -s`:

```bash
git commit -s -m "Describe the change"
```

This appends a line like:

```text
Signed-off-by: Your Name <you@example.com>
```

See [DCO.md](DCO.md) for the full certificate text.

## Development Workflow

1. Open an issue or discussion for substantial behavior, schema, or API
   changes before implementation.
2. Keep changes focused. Avoid unrelated refactors in the same pull request.
3. Add or update tests for behavior changes.
4. Update documentation when changing public APIs, schemas, examples, or
   operational behavior.
5. Run the relevant test suite before submitting.

For the full local test suite:

```bash
python -m pytest -q
```

For quick AMOS source imports without installing the package:

```bash
PYTHONPATH=src python -m pytest -q
```

## Project Boundaries

AMOS owns canonical memory, recall, provenance, maintenance, and memory packet
rendering. Integrations should keep external control authority, live execution,
approval checks, and domain-specific actuation outside AMOS unless the design
spec explicitly moves that boundary.

Memory maintenance should preserve provenance and auditability. Prefer typed
atoms, evidence links, explicit graph edges, and journaled maintenance actions
over prompt-only summaries.

## Code Style

- Prefer small, explicit functions over hidden global behavior.
- Keep deterministic maintenance paths independent of LLM calls.
- Preserve append-only journal semantics and replay verification for canonical
  state changes.
- Keep schema changes backward readable when practical, and document migration
  expectations when not.
- Avoid committing generated caches or local runtime stores.

## Reporting Security Issues

Please do not open public issues for sensitive security reports. Contact the
maintainers privately or open a minimal issue asking for a private reporting
channel.
