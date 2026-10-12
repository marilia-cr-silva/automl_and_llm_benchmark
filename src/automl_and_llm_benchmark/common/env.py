"""Environment-variable helpers.

An unset variable and a variable set to the empty string are treated alike, because the
stage Dockerfiles declare placeholders such as ``ENV VAR=""``. Unlike
``str(os.getenv("X"))``, which would silently produce the string ``"None"``, these helpers
never turn an unset variable into a string.
"""

from __future__ import annotations

import os


class MissingEnvVarError(RuntimeError):
    """Raised when a required environment variable is unset or empty."""


def get_env(name: str, default: str | None = None) -> str | None:
    """Return the variable's value, or ``default`` when it is unset or empty."""
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value


def require_env(name: str) -> str:
    """Return the variable's value or raise :class:`MissingEnvVarError`."""
    value = get_env(name)
    if value is None:
        raise MissingEnvVarError(
            f"Required environment variable {name} is not set (or is empty). "
            f"Set it with `docker run -e {name}=...` or export it; see docs/stages.md "
            "for the variables each stage needs."
        )
    return value


def _to_int(name: str, value: str) -> int:
    try:
        return int(value.strip())
    except ValueError:
        raise ValueError(
            f"Environment variable {name} must be an integer, got {value!r}."
        ) from None


def get_int_env(name: str, default: int | None = None) -> int | None:
    """Return the variable parsed as ``int``, or ``default`` when unset/empty."""
    value = get_env(name)
    if value is None:
        return default
    return _to_int(name, value)


def require_int_env(name: str) -> int:
    """Return the variable parsed as ``int`` or raise :class:`MissingEnvVarError`."""
    return _to_int(name, require_env(name))
