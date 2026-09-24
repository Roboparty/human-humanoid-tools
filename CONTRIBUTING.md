# Contributing to hhtools

Thank you for your interest in contributing. This project is in early alpha; feedback, bug reports,
and pull requests are all welcome.

## Development setup

```bash
git clone https://github.com/Roboparty/human-humanoid-tools
cd human-humanoid-tools
uv sync --extra all --extra dev
```

Run the linter:

```bash
ruff check hhtools
ruff format --check hhtools
```

## Code style

- Python 3.12+, `ruff` for linting and formatting (`line-length = 100`).
- Type-annotated public APIs.
- Every new module includes a module-level docstring.
- User-facing documentation lives in `README.md` (亮点与用法) and `framework.md` (框架与原理).

## Commit messages

Follow [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/):

```
feat(io): add Motion-X 322-dim adapter
fix(retarget): stabilise IK warm-start for short clips
docs: update framework.md
```

## License

By contributing, you agree that your contributions will be licensed under the
[Apache-2.0](LICENSE) license. Do not commit third-party datasets, SMPL weights,
or other assets whose license prohibits redistribution.
