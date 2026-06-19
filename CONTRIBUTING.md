# Contributing to QuickPCA

Thanks for your interest in improving QuickPCA! Contributions of all kinds are
welcome — bug reports, documentation, tests, and code.

## Development setup

```bash
git clone https://github.com/5x5x5x5/quickpca
cd quickpca
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Quality checks

Run the full suite locally before opening a pull request:

```bash
pytest                 # tests
ruff check .           # lint
ruff format --check .  # formatting
mypy quickpca          # type checking
```

`pre-commit` runs the lint/format hooks automatically on each commit.

## Pull request conventions

- Branch from the default branch and keep PRs focused on a single change.
- Write a clear, imperative title (e.g. "Add Cython backend").
- Include or update tests for behavioural changes.
- Ensure `pytest`, `ruff check`, `ruff format --check`, and `mypy` all pass.
- Update docs and `CHANGELOG.md` (under an `Unreleased` heading) when relevant.

## Adding a backend

Backends self-register via the `register_backend` decorator. Create a module in
`quickpca/backends/` and decorate a `Backend` subclass that sets a unique
`name`:

```python
from quickpca.backends import register_backend
from quickpca.backends.base import Backend

@register_backend
class MyBackend(Backend):
    name = "mybackend"
    # implement the abstract methods from Backend
```

Importing the module registers the backend; `quickpca backends` lists all
registered names.

## Adding a CLI subcommand

Drop a module into `quickpca/commands/`. It is auto-discovered at startup and
must expose a `register(subparsers)` function that adds a parser and wires a
handler via `set_defaults(func=...)`, where `func` is a callable taking the
parsed `args` and returning an `int` exit code:

```python
def _run(args):
    ...
    return 0

def register(subparsers):
    p = subparsers.add_parser("mycmd", help="Describe the command.")
    p.add_argument("--option")
    p.set_defaults(func=_run)
```

## Code of conduct

By participating you agree to abide by the [Code of Conduct](CODE_OF_CONDUCT.md).
