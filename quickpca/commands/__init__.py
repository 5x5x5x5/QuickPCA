"""Auto-discovered CLI subcommand package.

Worker-contributed subcommand modules are dropped into this package. Each must
define ``def register(subparsers)`` which adds a parser and sets ``func`` via
``set_defaults(func=...)`` to a callable ``(args) -> int``.
"""
