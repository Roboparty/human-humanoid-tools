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

## Frontend development

The WebUI and Electron GUI use one React + TypeScript renderer from `hhtools/web/frontend`;
Electron loads the same page through its local FastAPI sidecar, so there is no second GUI renderer.
Tailwind CSS is wired into the build, and shadcn/ui primitives are copied into the project only
when a real view needs them.

```bash
cd hhtools/web/frontend
npm install
npm run typecheck
npm test
npm run build
```

The production build is written to `hhtools/web/static`, which is served unchanged by both FastAPI
and Electron. Desktop packaging is covered in [`desktop/README.md`](desktop/README.md).

## Code style

- Python 3.12+, `ruff` for linting and formatting (`line-length = 100`).
- Type-annotated public APIs.
- Every new module includes a module-level docstring.
- User-facing documentation lives in `README.md` (亮点与快速开始), the guides under `docs/`
  (安装、CLI、机器人调参、数据集等), and `framework.md` (框架与原理).

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
