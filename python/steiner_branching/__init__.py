"""Independent research stack for learning to branch on Steiner problems."""

from .config import ScaffoldConfig, StrictConfigError, load_dataclass_yaml
from .contracts import (
    EdgeVariableMetadata,
    GraphSchema,
    ProblemMetadata,
    RunManifest,
    SteinerEdge,
    SteinerGraph,
)
from .runtime import ArtifactLayout, configure_logging, seed_everything

__all__ = [
    "ArtifactLayout",
    "EdgeVariableMetadata",
    "GraphSchema",
    "ProblemMetadata",
    "RunManifest",
    "ScaffoldConfig",
    "SteinerEdge",
    "SteinerGraph",
    "StrictConfigError",
    "configure_logging",
    "load_dataclass_yaml",
    "seed_everything",
]

__version__ = "0.1.0"
