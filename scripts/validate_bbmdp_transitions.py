#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Validate BBMDP smoke episode records")
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--repeat-pair", nargs=2, type=Path)
    return parser.parse_args()


def stable_episode(path):
    episode = json.loads(path.read_text())
    episode.pop("solving_time", None)
    return episode


def validate(path):
    episode = json.loads(path.read_text())
    transitions = episode["transitions"]
    errors = []
    if len(transitions) != episode["transition_count"]:
        errors.append("transition count mismatch")
    if episode["gamma"] != 1.0:
        errors.append("gamma is not one")
    for transition in transitions:
        if not 0 <= transition["action_position"] < transition["candidate_count"]:
            errors.append(f"step {transition['step']}: action position outside action mask")
        if transition["variable_shape"][1] != 19 or transition["constraint_shape"][1] != 5:
            errors.append(f"step {transition['step']}: invalid NodeBipartite feature shape")
        if episode["reward_mode"] == "negative_node_increment":
            expected = -(transition["nodes_after"] - transition["nodes_before"])
            if transition["reward"] != expected:
                errors.append(f"step {transition['step']}: node-increment reward mismatch")
        elif transition["reward"] != -1.0:
            errors.append(f"step {transition['step']}: constant reward mismatch")
        if transition["next_variable_shape"] is not None:
            if transition["next_variable_shape"][0] != transition["variable_shape"][0]:
                errors.append(f"step {transition['step']}: variable dimension changed")
            if transition["next_variable_shape"][1] != 19:
                errors.append(f"step {transition['step']}: next variable width mismatch")
            if transition["next_constraint_shape"][1] != 5:
                errors.append(f"step {transition['step']}: next constraint width mismatch")
    if transitions:
        final = transitions[-1]
        if not (final["terminated"] or final["truncated"]):
            errors.append("last transition is not final")
        if final["bootstrap_mask"] != 0.0:
            errors.append("final bootstrap mask is not zero")
    if errors:
        raise ValueError(f"{path}: " + "; ".join(errors))
    return len(transitions)


def main():
    args = parse_args()
    total = sum(validate(path) for path in args.episodes)
    if args.repeat_pair and stable_episode(args.repeat_pair[0]) != stable_episode(args.repeat_pair[1]):
        raise SystemExit("repeat episodes differ after removing solving_time")
    print(f"validated {len(args.episodes)} episodes and {total} transitions")


if __name__ == "__main__":
    main()
