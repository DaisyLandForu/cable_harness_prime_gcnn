"""Fail-closed SCIP probindex identity for Ecole variable rows."""

from __future__ import annotations

from collections.abc import Callable, Iterable
import ctypes
from functools import lru_cache
import hashlib
import numbers
import os
from pathlib import Path
from typing import Any


EXPECTED_STACK_ID = "scip804-ecole081-pyscipopt430"
EXPECTED_SCIP_VERSION = "8.0.4"
EXPECTED_LIBSCIP_SHA256 = (
    "5524e92770f25c1baa6c1469528a71fadcc25aeb0db585b0938030ec857281ee"
)
REPO = Path(__file__).resolve().parents[3]
FROZEN_SCIP_PREFIX = REPO / "artifacts/environment/phase4/scip804_prefix"
FROZEN_LIBSCIP = FROZEN_SCIP_PREFIX / "lib/libscip.so.8.0"


class ScipIdentityError(RuntimeError):
    """Raised when SCIP variable identity cannot be proven exactly."""


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def _probindex_symbol() -> tuple[Any, Any]:
    """Load SCIPvarGetProbindex only from the wrapper-verified SCIP 8.0.4."""
    if os.environ.get("STEINER_SOLVER_STACK_ID") != EXPECTED_STACK_ID:
        raise ScipIdentityError("SCIP probindex access requires the frozen Steiner wrapper")
    if os.environ.get("STEINER_SCIP_VERSION") != EXPECTED_SCIP_VERSION:
        raise ScipIdentityError("SCIP probindex access requires SCIP 8.0.4")
    prefix_value = os.environ.get("SCIPOPTDIR")
    if not prefix_value:
        raise ScipIdentityError("SCIPOPTDIR is missing from the frozen Steiner wrapper")
    try:
        observed_prefix = Path(prefix_value).resolve(strict=True)
        expected_prefix = FROZEN_SCIP_PREFIX.resolve(strict=True)
    except OSError as error:
        raise ScipIdentityError("the frozen SCIP prefix cannot be resolved") from error
    if observed_prefix != expected_prefix:
        raise ScipIdentityError(
            f"SCIPOPTDIR is not the repository-frozen prefix: {observed_prefix}"
        )
    if not FROZEN_LIBSCIP.is_file():
        raise ScipIdentityError(f"the frozen libscip is missing: {FROZEN_LIBSCIP}")
    actual_sha256 = _file_sha256(FROZEN_LIBSCIP)
    if actual_sha256 != EXPECTED_LIBSCIP_SHA256:
        raise ScipIdentityError(
            "the frozen libscip checksum changed: "
            f"expected={EXPECTED_LIBSCIP_SHA256} actual={actual_sha256}"
        )
    try:
        library = ctypes.CDLL(str(FROZEN_LIBSCIP.resolve(strict=True)))
        function = library.SCIPvarGetProbindex
    except (OSError, AttributeError) as error:
        raise ScipIdentityError(
            "SCIPvarGetProbindex is unavailable in the frozen SCIP 8.0.4 library"
        ) from error
    function.argtypes = [ctypes.c_void_p]
    function.restype = ctypes.c_int
    return library, function


def scip_variable_probindex(variable: Any) -> int:
    """Read an active PySCIPOpt variable's problem-array index via SCIP C API."""
    try:
        pointer = variable.ptr()
    except (AttributeError, TypeError) as error:
        raise ScipIdentityError("PySCIPOpt variable does not expose a SCIP pointer") from error
    if isinstance(pointer, bool) or not isinstance(pointer, numbers.Integral) or pointer <= 0:
        raise ScipIdentityError("PySCIPOpt variable returned an invalid SCIP pointer")
    _library, function = _probindex_symbol()
    return int(function(ctypes.c_void_p(int(pointer))))


def variable_names_by_probindex(
    variables: Iterable[Any],
    row_count: int,
    *,
    probindex_of: Callable[[Any], int] | None = None,
) -> tuple[str, ...]:
    """Bind names to Ecole rows using a complete probindex bijection.

    Ecole's variable row ``i`` is SCIP probindex ``i``. The input iterable may
    arrive in any order; list position is deliberately ignored.
    """
    if isinstance(row_count, bool) or not isinstance(row_count, numbers.Integral):
        raise ScipIdentityError("Ecole variable row count must be an integer")
    n_rows = int(row_count)
    if n_rows < 0:
        raise ScipIdentityError("Ecole variable row count must be non-negative")
    variable_list = tuple(variables)
    if len(variable_list) != n_rows:
        raise ScipIdentityError(
            f"SCIP/Ecole variable count mismatch: scip={len(variable_list)} ecole={n_rows}"
        )
    read_probindex = probindex_of or scip_variable_probindex
    names: list[str | None] = [None] * n_rows
    seen_names: set[str] = set()
    for variable in variable_list:
        raw_probindex = read_probindex(variable)
        if isinstance(raw_probindex, bool) or not isinstance(raw_probindex, numbers.Integral):
            raise ScipIdentityError("SCIP variable probindex must be an integer")
        probindex = int(raw_probindex)
        if not 0 <= probindex < n_rows:
            raise ScipIdentityError(
                f"SCIP variable probindex is outside Ecole rows: {probindex}/{n_rows}"
            )
        if names[probindex] is not None:
            raise ScipIdentityError(f"duplicate SCIP variable probindex: {probindex}")
        raw_name = getattr(variable, "name", None)
        if not isinstance(raw_name, str) or not raw_name or raw_name in seen_names:
            raise ScipIdentityError(
                f"missing or duplicate SCIP variable name: {raw_name!r}"
            )
        name = raw_name
        names[probindex] = name
        seen_names.add(name)
    missing = [index for index, name in enumerate(names) if name is None]
    if missing:
        raise ScipIdentityError(f"missing SCIP variable probindices: {missing[:20]}")
    return tuple(str(name) for name in names)


def transformed_variable_names_by_probindex(
    pyscip_model: Any, row_count: int
) -> tuple[str, ...]:
    """Return transformed names in canonical Ecole/probindex row order."""
    try:
        variables = pyscip_model.getVars(transformed=True)
    except (AttributeError, TypeError) as error:
        raise ScipIdentityError("PySCIPOpt model cannot return transformed variables") from error
    return variable_names_by_probindex(variables, row_count)
