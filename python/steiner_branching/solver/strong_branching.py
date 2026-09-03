"""Frozen SCIP 8.0.4 strong-branch labels with explicit child validity."""

from __future__ import annotations

import ctypes
from functools import lru_cache
import math
import time
from typing import Any

import ecole
import numpy as np

from ..learning.teacher_data import StrongBranchLabels, immutable_array
from .scip_identity import frozen_scip_library, scip_variable_probindex


class StrongBranchingError(RuntimeError):
    """Raised when a complete strong-branch label cannot be obtained safely."""


def _check_retcode(code: int, operation: str) -> None:
    # SCIP 8.x deliberately defines SCIP_OKAY as +1, not the conventional zero.
    if int(code) != 1:
        raise StrongBranchingError(f"{operation} failed with SCIP_RETCODE={int(code)}")


@lru_cache(maxsize=1)
def _strong_api() -> dict[str, Any]:
    library = frozen_scip_library()
    void = ctypes.c_void_p
    real = ctypes.c_double
    boolean = ctypes.c_uint

    library.SCIPstartStrongbranch.argtypes = [void, boolean]
    library.SCIPstartStrongbranch.restype = ctypes.c_int
    library.SCIPendStrongbranch.argtypes = [void]
    library.SCIPendStrongbranch.restype = ctypes.c_int
    library.SCIPgetVarStrongbranchFrac.argtypes = [
        void, void, ctypes.c_int, boolean,
        ctypes.POINTER(real), ctypes.POINTER(real),
        ctypes.POINTER(boolean), ctypes.POINTER(boolean),
        ctypes.POINTER(boolean), ctypes.POINTER(boolean),
        ctypes.POINTER(boolean), ctypes.POINTER(boolean), ctypes.POINTER(boolean),
    ]
    library.SCIPgetVarStrongbranchFrac.restype = ctypes.c_int
    library.SCIPgetBranchScore.argtypes = [void, void, real, real]
    library.SCIPgetBranchScore.restype = real
    library.SCIPgetLPObjval.argtypes = [void]
    library.SCIPgetLPObjval.restype = real
    for name in (
        "SCIPgetNLPIterations",
        "SCIPgetNStrongbranchLPIterations",
        "SCIPgetNStrongbranchs",
    ):
        function = getattr(library, name)
        function.argtypes = [void]
        function.restype = ctypes.c_longlong
    return {"library": library}


def _model_pointer(pyscip_model: Any) -> tuple[Any, ctypes.c_void_p]:
    try:
        capsule = pyscip_model.to_ptr(False)
    except (AttributeError, TypeError) as error:
        raise StrongBranchingError("PySCIPOpt model does not expose its SCIP capsule") from error
    get_pointer = ctypes.pythonapi.PyCapsule_GetPointer
    get_pointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
    get_pointer.restype = ctypes.c_void_p
    pointer = get_pointer(capsule, b"scip")
    if not pointer:
        raise StrongBranchingError("PySCIPOpt returned a null SCIP pointer")
    return capsule, ctypes.c_void_p(pointer)


def _variable_pointer(variable: Any) -> ctypes.c_void_p:
    try:
        pointer = int(variable.ptr())
    except (AttributeError, TypeError, ValueError) as error:
        raise StrongBranchingError("candidate does not expose a SCIP_VAR pointer") from error
    if pointer <= 0:
        raise StrongBranchingError("candidate returned a null SCIP_VAR pointer")
    return ctypes.c_void_p(pointer)


class StrongBranchingTeacher:
    """Ecole observation function returning all legal LP-candidate labels."""

    def __init__(
        self,
        *,
        iteration_limit: int,
        candidate_limit: int = 0,
        idempotent: bool = False,
        infeasible_gain_cap: float = 1.0e6,
    ) -> None:
        if iteration_limit < 1 or candidate_limit < 0:
            raise ValueError("strong-branch limits are invalid")
        if not math.isfinite(infeasible_gain_cap) or infeasible_gain_cap <= 0.0:
            raise ValueError("infeasible_gain_cap must be finite and positive")
        self.iteration_limit = int(iteration_limit)
        self.candidate_limit = int(candidate_limit)
        self.idempotent = bool(idempotent)
        self.infeasible_gain_cap = float(infeasible_gain_cap)

    def before_reset(self, model: Any) -> None:
        del model

    def extract(self, model: Any, done: bool) -> StrongBranchLabels | None:
        if done:
            return None
        pyscip_model = model.as_pyscipopt()
        values = pyscip_model.getLPBranchCands()
        candidates = tuple(values[0][: int(values[4])])
        if not candidates:
            raise StrongBranchingError("Ecole requested a teacher outside a legal branch state")
        n_evaluate = (
            min(len(candidates), self.candidate_limit)
            if self.candidate_limit > 0
            else len(candidates)
        )
        if n_evaluate != len(candidates):
            raise StrongBranchingError(
                "S05 pilot forbids partial candidate labeling; candidate_limit must be zero"
            )
        probindices = np.asarray(
            [scip_variable_probindex(variable) for variable in candidates], dtype=np.int64
        )
        if len(set(map(int, probindices))) != len(candidates):
            raise StrongBranchingError("SCIP returned duplicate legal candidate probindices")

        api = _strong_api()["library"]
        capsule, scip = _model_pointer(pyscip_model)
        del capsule  # the PySCIPOpt model keeps the SCIP pointer alive for this extraction
        lp_objective = float(api.SCIPgetLPObjval(scip))
        if not math.isfinite(lp_objective):
            raise StrongBranchingError("current LP objective is non-finite")
        lp_before = int(api.SCIPgetNLPIterations(scip))
        strong_lp_before = int(api.SCIPgetNStrongbranchLPIterations(scip))
        calls_before = int(api.SCIPgetNStrongbranchs(scip))

        down_bounds: list[float] = []
        up_bounds: list[float] = []
        down_valid_flags: list[bool] = []
        up_valid_flags: list[bool] = []
        down_infeasible_flags: list[bool] = []
        up_infeasible_flags: list[bool] = []
        lp_error_flags: list[bool] = []
        scores: list[float] = []
        started = time.monotonic()
        _check_retcode(api.SCIPstartStrongbranch(scip, 0), "SCIPstartStrongbranch")
        pending_error: BaseException | None = None
        try:
            for variable in candidates:
                down = ctypes.c_double(0.0)
                up = ctypes.c_double(0.0)
                down_valid = ctypes.c_uint(0)
                up_valid = ctypes.c_uint(0)
                down_infeasible = ctypes.c_uint(0)
                up_infeasible = ctypes.c_uint(0)
                down_conflict = ctypes.c_uint(0)
                up_conflict = ctypes.c_uint(0)
                lp_error = ctypes.c_uint(0)
                _check_retcode(
                    api.SCIPgetVarStrongbranchFrac(
                        scip,
                        _variable_pointer(variable),
                        self.iteration_limit,
                        int(self.idempotent),
                        ctypes.byref(down),
                        ctypes.byref(up),
                        ctypes.byref(down_valid),
                        ctypes.byref(up_valid),
                        ctypes.byref(down_infeasible),
                        ctypes.byref(up_infeasible),
                        ctypes.byref(down_conflict),
                        ctypes.byref(up_conflict),
                        ctypes.byref(lp_error),
                    ),
                    "SCIPgetVarStrongbranchFrac",
                )
                down_ok = bool(down_valid.value or down_infeasible.value)
                up_ok = bool(up_valid.value or up_infeasible.value)
                fully_valid = not lp_error.value and down_ok and up_ok
                score = math.nan
                if fully_valid:
                    down_gain = (
                        self.infeasible_gain_cap
                        if down_infeasible.value
                        else max(float(down.value) - lp_objective, 0.0)
                    )
                    up_gain = (
                        self.infeasible_gain_cap
                        if up_infeasible.value
                        else max(float(up.value) - lp_objective, 0.0)
                    )
                    score = float(
                        api.SCIPgetBranchScore(
                            scip, _variable_pointer(variable), down_gain, up_gain
                        )
                    )
                    if not math.isfinite(score):
                        raise StrongBranchingError("SCIP produced a non-finite valid score")
                down_bounds.append(float(down.value))
                up_bounds.append(float(up.value))
                down_valid_flags.append(bool(down_valid.value))
                up_valid_flags.append(bool(up_valid.value))
                down_infeasible_flags.append(bool(down_infeasible.value))
                up_infeasible_flags.append(bool(up_infeasible.value))
                lp_error_flags.append(bool(lp_error.value))
                scores.append(score)
        except BaseException as error:
            pending_error = error
        finally:
            try:
                _check_retcode(api.SCIPendStrongbranch(scip), "SCIPendStrongbranch")
            except BaseException as end_error:
                if pending_error is None:
                    pending_error = end_error
        if pending_error is not None:
            raise pending_error
        elapsed = time.monotonic() - started
        return StrongBranchLabels(
            candidate_probindices=immutable_array(probindices, np.int64),
            scores=immutable_array(scores, np.float64),
            down_bounds=immutable_array(down_bounds, np.float64),
            up_bounds=immutable_array(up_bounds, np.float64),
            down_valid=immutable_array(down_valid_flags, np.bool_),
            up_valid=immutable_array(up_valid_flags, np.bool_),
            down_infeasible=immutable_array(down_infeasible_flags, np.bool_),
            up_infeasible=immutable_array(up_infeasible_flags, np.bool_),
            lp_errors=immutable_array(lp_error_flags, np.bool_),
            node_number=int(pyscip_model.getCurrentNode().getNumber()),
            depth=int(pyscip_model.getCurrentNode().getDepth()),
            elapsed_seconds=elapsed,
            lp_iterations_delta=int(api.SCIPgetNLPIterations(scip)) - lp_before,
            strong_lp_iterations_delta=(
                int(api.SCIPgetNStrongbranchLPIterations(scip)) - strong_lp_before
            ),
            strong_calls_delta=int(api.SCIPgetNStrongbranchs(scip)) - calls_before,
        )


class OrderedStrongBranchObservation:
    """Extract state/pseudocost before teacher updates, in an explicit order."""

    def __init__(
        self,
        state_observation: Any,
        teacher_observation: StrongBranchingTeacher,
        *,
        include_ecole_reference: bool = False,
    ) -> None:
        self.state_observation = state_observation
        self.pseudocost_observation = ecole.observation.Pseudocosts()
        self.teacher_observation = teacher_observation
        self.reference_observation = (
            ecole.observation.StrongBranchingScores(pseudo_candidates=False)
            if include_ecole_reference
            else None
        )

    def before_reset(self, model: Any) -> None:
        self.state_observation.before_reset(model)
        self.pseudocost_observation.before_reset(model)
        self.teacher_observation.before_reset(model)
        if self.reference_observation is not None:
            self.reference_observation.before_reset(model)

    def extract(self, model: Any, done: bool) -> dict[str, Any]:
        result = {
            "state": self.state_observation.extract(model, done),
            "pseudocosts": self.pseudocost_observation.extract(model, done),
            "teacher": self.teacher_observation.extract(model, done),
        }
        if self.reference_observation is not None:
            result["ecole_teacher"] = self.reference_observation.extract(model, done)
        return result


def align_labels_to_probindices(
    labels: StrongBranchLabels, target_probindices: Any
) -> StrongBranchLabels:
    """Align candidate arrays by identity, rejecting missing or extra actions."""
    target = np.asarray(target_probindices, dtype=np.int64)
    if target.ndim != 1 or len(set(map(int, target))) != target.size:
        raise StrongBranchingError("Ecole action_set is not a unique one-dimensional list")
    positions = {int(value): index for index, value in enumerate(labels.candidate_probindices)}
    if set(positions) != set(map(int, target)):
        raise StrongBranchingError("teacher candidates and Ecole action_set are not a bijection")
    order = np.asarray([positions[int(value)] for value in target], dtype=np.int64)
    return StrongBranchLabels(
        candidate_probindices=immutable_array(target, np.int64),
        scores=immutable_array(labels.scores[order], np.float64),
        down_bounds=immutable_array(labels.down_bounds[order], np.float64),
        up_bounds=immutable_array(labels.up_bounds[order], np.float64),
        down_valid=immutable_array(labels.down_valid[order], np.bool_),
        up_valid=immutable_array(labels.up_valid[order], np.bool_),
        down_infeasible=immutable_array(labels.down_infeasible[order], np.bool_),
        up_infeasible=immutable_array(labels.up_infeasible[order], np.bool_),
        lp_errors=immutable_array(labels.lp_errors[order], np.bool_),
        node_number=labels.node_number,
        depth=labels.depth,
        elapsed_seconds=labels.elapsed_seconds,
        lp_iterations_delta=labels.lp_iterations_delta,
        strong_lp_iterations_delta=labels.strong_lp_iterations_delta,
        strong_calls_delta=labels.strong_calls_delta,
    )
