"""Same-interpreter command entry point for ``python -m evalt``."""

from .cli import main


if __name__ == "__main__":
    raise SystemExit(main())
