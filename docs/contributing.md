# Contributing

Contributions to QuickPCA are welcome — bug reports, fixes, new backends,
additional CLI subcommands, and documentation improvements alike. This page
covers the local development workflow.

## Development setup

Clone the repository and install it in **editable** mode with the `dev` extra:

```bash
git clone https://github.com/5x5x5x5/quickpca.git
cd quickpca
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

The `dev` extra pulls in `pytest`, `pytest-cov`, `ruff`, `mypy`, and
`pre-commit`. To work on the accelerated backend or the docs, add the relevant
extras:

```bash
pip install -e ".[dev,jax,docs]"
```

## Running the tests

The test suite lives under `tests/` and is configured through `pyproject.toml`:

```bash
pytest
```

Run with coverage for the `quickpca` package:

```bash
pytest --cov=quickpca --cov-report=term-missing
```

## Linting and type checking

QuickPCA uses [Ruff](https://docs.astral.sh/ruff/) for linting and import
sorting, and [mypy](https://mypy-lang.org/) for type checking. The Ruff
configuration (rule set `E`, `F`, `I`, `UP`, `B`; line length 100) lives in
`pyproject.toml`.

```bash
ruff check .          # lint
ruff check --fix .    # auto-fix where possible
ruff format .         # format
mypy quickpca         # type-check
```

### Pre-commit

Install the git hooks so linting runs automatically on every commit:

```bash
pre-commit install
pre-commit run --all-files
```

## Building the documentation

This site is built with [MkDocs Material](https://squidfunk.github.io/mkdocs-material/)
and [mkdocstrings](https://mkdocstrings.github.io/). Install the `docs` extra,
then serve or build:

```bash
pip install -e ".[docs]"
mkdocs serve            # live-reloading preview at http://127.0.0.1:8000
mkdocs build --strict   # production build; warnings are treated as errors
```

Because `mkdocstrings` imports the package to read docstrings, keep `quickpca`
importable and make sure new public symbols are documented and added to the
[API reference](api.md).

## Adding a CLI subcommand

The CLI auto-discovers subcommands from the `quickpca.commands` package. To add
one, drop a module there that defines a `register(subparsers)` function:

```python
# quickpca/commands/mycommand.py
def register(subparsers):
    p = subparsers.add_parser("mycommand", help="What it does.")
    p.add_argument("input")
    p.set_defaults(func=_run)

def _run(args):
    ...
    return 0  # process exit code
```

It will appear automatically in `quickpca --help`. A module that fails to import
or register is reported as a warning and skipped, so it can never break the rest
of the CLI. See the [CLI guide](cli.md) for details.

## Adding a backend

Subclass `Backend`, set a unique `name`, implement `align_frames` and `pca`, and
register the class with the `@register_backend` decorator. See the
[Backends](backends.md) guide and the [API reference](api.md#quickpca.backends.base.Backend)
for the full protocol.

## Pull requests

1. Branch off the development branch.
2. Add or update tests for your change.
3. Ensure `pytest`, `ruff check`, and `mypy` all pass.
4. Keep the public API documented and update the docs where relevant.

## License & attribution

QuickPCA was created by **Gleb Novikov** (The Visual Hub) and is released under
the **MIT License**. By contributing you agree that your contributions are
licensed under the same terms.
