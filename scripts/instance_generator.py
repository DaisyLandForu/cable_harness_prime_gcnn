#!/usr/bin/env python3
import argparse
import csv
import hashlib
import itertools
import json
import random
from pathlib import Path


PAIR_WEIGHTS = (1e-7, 2e-7, 4e-7)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def generate_instance(instance_id, seed, parameters, output_dir):
    rng = random.Random(seed)
    backbone_count = int(parameters["backbone_nodes"])
    entry_count = int(parameters["entry_nodes"])
    extra_count = int(parameters["extra_backbone_edges"])
    cable_count = int(parameters["cables"])
    if backbone_count < 2 or entry_count < 2 or cable_count < 1:
        raise ValueError("generator counts are too small")
    if cable_count > entry_count * (entry_count - 1) // 2:
        raise ValueError("not enough unique entry-node pairs for requested cables")

    scenario = f"SYN_{instance_id.upper()}"
    backbone = [f"N{index}" for index in range(backbone_count)]
    entries = [f"E{index}" for index in range(entry_count)]
    edges = []
    used_edges = set()

    def add_edge(source, target, edge_class, low, high):
        key = tuple(sorted((source, target)))
        if key in used_edges:
            return False
        used_edges.add(key)
        edges.append(
            {
                "source": source,
                "target": target,
                "weight": rng.randint(low, high),
                "class": edge_class,
            }
        )
        return True

    for index in range(backbone_count - 1):
        add_edge(backbone[index], backbone[index + 1], "backbone", 80, 500)

    candidates = [
        pair
        for pair in itertools.combinations(backbone, 2)
        if tuple(sorted(pair)) not in used_edges
    ]
    rng.shuffle(candidates)
    if extra_count > len(candidates):
        raise ValueError("too many extra backbone edges")
    for source, target in candidates[:extra_count]:
        add_edge(source, target, "backbone_extra", 80, 500)

    for entry in entries:
        add_edge(entry, rng.choice(backbone), "entry", 30, 250)

    entry_pairs = list(itertools.combinations(entries, 2))
    rng.shuffle(entry_pairs)
    selected_entry_pairs = entry_pairs[:cable_count]
    pairs = []
    for index, (entry_a, entry_b) in enumerate(selected_entry_pairs):
        leaf_a = f"L{index:04d}A"
        leaf_b = f"L{index:04d}B"
        add_edge(leaf_a, entry_a, "access", 10, 100)
        add_edge(leaf_b, entry_b, "access", 10, 100)
        pairs.append(
            {
                "source": leaf_a,
                "target": leaf_b,
                "weight": PAIR_WEIGHTS[index % len(PAIR_WEIGHTS)],
                "entry_source": entry_a,
                "entry_target": entry_b,
            }
        )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges_path = output_dir / f"{instance_id}_edges.csv"
    pairs_path = output_dir / f"{instance_id}_pairs.csv"
    with edges_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "scenario", "source", "target", "length"])
        for index, edge in enumerate(edges, start=1):
            writer.writerow([index, scenario, edge["source"], edge["target"], edge["weight"]])
    with pairs_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(["id", "scenario", "cable_id", "source", "target", "weight"])
        for index, pair in enumerate(pairs, start=1):
            writer.writerow(
                [index, scenario, f"CABLE-{index:04d}", pair["source"], pair["target"], pair["weight"]]
            )

    graph_nodes = set()
    for edge in edges:
        graph_nodes.add(edge["source"])
        graph_nodes.add(edge["target"])
    metadata = {
        "instance_id": instance_id,
        "scenario": scenario,
        "seed": seed,
        "parameters": dict(parameters),
        "graph": {
            "nodes": len(graph_nodes),
            "edges": len(edges),
            "center_nodes": backbone_count + entry_count,
            "center_edges": backbone_count - 1 + extra_count + entry_count,
            "access_edges": 2 * cable_count,
            "cables": cable_count,
        },
        "semantics": {
            "backbone_nodes": "N followed by digits",
            "entry_nodes": "E followed by digits",
            "leaf_nodes": "L-prefixed cable endpoints",
            "pair_weight_schedule": list(PAIR_WEIGHTS),
        },
        "source_files": {
            "edges": str(edges_path),
            "pairs": str(pairs_path),
            "edges_sha256": sha256_file(edges_path),
            "pairs_sha256": sha256_file(pairs_path),
        },
    }
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Generate a deterministic cable-harness MILP source instance")
    parser.add_argument("--instance-id", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--backbone-nodes", type=int, required=True)
    parser.add_argument("--entry-nodes", type=int, required=True)
    parser.add_argument("--extra-backbone-edges", type=int, required=True)
    parser.add_argument("--cables", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--metadata-json")
    args = parser.parse_args()
    parameters = {
        "backbone_nodes": args.backbone_nodes,
        "entry_nodes": args.entry_nodes,
        "extra_backbone_edges": args.extra_backbone_edges,
        "cables": args.cables,
    }
    metadata = generate_instance(args.instance_id, args.seed, parameters, args.output_dir)
    if args.metadata_json:
        path = Path(args.metadata_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(metadata, sort_keys=True))


if __name__ == "__main__":
    main()
