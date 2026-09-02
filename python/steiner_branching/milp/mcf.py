"""Rooted multi-commodity-flow formulation for classic undirected SPG."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..contracts import EdgeVariableMetadata, ProblemMetadata, SteinerGraph
from .naming import edge_variable_name, flow_variable_name


@dataclass(frozen=True)
class McfCounts:
    binary_edge_variables: int
    continuous_flow_variables: int
    flow_balance_constraints: int
    linking_constraints: int


@dataclass
class McfBuild:
    graph: SteinerGraph
    model: Any
    edge_variables: dict[int, Any]
    flow_variables: dict[tuple[int, int], Any]
    metadata: ProblemMetadata
    counts: McfCounts


def configure_p0(model: Any, *, seed: int = 0) -> None:
    if seed < 0:
        raise ValueError("seed must be non-negative")
    model.setParam("limits/time", 60.0)
    model.setParam("limits/nodes", 10_000)
    model.setParam("limits/memory", 4096.0)
    model.setParam("parallel/minnthreads", 1)
    model.setParam("parallel/maxnthreads", 1)
    model.setParam("lp/threads", 1)
    model.setParam("randomization/randomseedshift", seed)
    model.setParam("randomization/permutationseed", seed)
    model.setParam("randomization/lpseed", seed)


def build_mcf(
    graph: SteinerGraph, *, configure_correctness_profile: bool = True, hide_output: bool = True
) -> McfBuild:
    try:
        from pyscipopt import Model, quicksum
    except ImportError as error:
        raise RuntimeError(
            "PySCIPOpt is required for MCF construction; activate the locked rl4scip environment"
        ) from error
    model = Model(f"spg-mcf-{graph.name}")
    if hide_output:
        model.hideOutput()
    if configure_correctness_profile:
        configure_p0(model, seed=0)
    edge_variables = {
        edge.edge_id: model.addVar(
            name=edge_variable_name(edge.edge_id), vtype="B", lb=0.0, ub=1.0, obj=edge.cost
        )
        for edge in graph.edges
    }
    arcs: list[tuple[int, int, int, int]] = []
    for edge in graph.edges:
        arcs.append((2 * edge.edge_id, edge.u, edge.v, edge.edge_id))
        arcs.append((2 * edge.edge_id + 1, edge.v, edge.u, edge.edge_id))
    commodities = tuple(terminal for terminal in graph.terminals if terminal != graph.root)
    flow_variables: dict[tuple[int, int], Any] = {}
    for terminal in commodities:
        for arc_id, _, _, _ in arcs:
            flow_variables[(terminal, arc_id)] = model.addVar(
                name=flow_variable_name(terminal, arc_id),
                vtype="C",
                lb=0.0,
                ub=1.0,
                obj=0.0,
            )
    balance_count = 0
    link_count = 0
    for terminal in commodities:
        for node in graph.nodes:
            outgoing = [
                flow_variables[(terminal, arc_id)]
                for arc_id, source, _, _ in arcs
                if source == node
            ]
            incoming = [
                flow_variables[(terminal, arc_id)]
                for arc_id, _, target, _ in arcs
                if target == node
            ]
            rhs = 1.0 if node == graph.root else -1.0 if node == terminal else 0.0
            model.addCons(
                quicksum(outgoing) - quicksum(incoming) == rhs,
                name=f"stp_flowbal_t{terminal:04d}_v{node:08d}",
            )
            balance_count += 1
        for edge in graph.edges:
            forward = flow_variables[(terminal, 2 * edge.edge_id)]
            backward = flow_variables[(terminal, 2 * edge.edge_id + 1)]
            model.addCons(
                forward + backward <= edge_variables[edge.edge_id],
                name=f"stp_link_t{terminal:04d}_e{edge.edge_id:08d}",
            )
            link_count += 1
    model.setMinimize()
    metadata = ProblemMetadata(
        schema_version=1,
        problem="SPG",
        formulation_id="rooted_mcf_v1",
        formulation_version=1,
        instance_name=graph.name,
        source_sha256=graph.source_sha256,
        graph_sha256=graph.graph_sha256,
        root=graph.root,
        terminals=graph.terminals,
        edge_variables=tuple(
            EdgeVariableMetadata(
                edge_id=edge.edge_id,
                u=edge.u,
                v=edge.v,
                variable_name=edge_variable_name(edge.edge_id),
            )
            for edge in graph.edges
        ),
    )
    return McfBuild(
        graph=graph,
        model=model,
        edge_variables=edge_variables,
        flow_variables=flow_variables,
        metadata=metadata,
        counts=McfCounts(
            binary_edge_variables=len(graph.edges),
            continuous_flow_variables=2 * len(graph.edges) * len(commodities),
            flow_balance_constraints=len(graph.nodes) * len(commodities),
            linking_constraints=len(graph.edges) * len(commodities),
        ),
    )
