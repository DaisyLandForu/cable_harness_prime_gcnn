import argparse
import random
import csv
from pathlib import Path

# edges = pd.read_csv('edges.csv', header=None, names=['source', 'target'])
# pairs = pd.read_csv('pairs.csv', header=None, names=['node1', 'node2'])

def generate_connected_graph(n, m, k, num_center, rng=None):
    rng = rng or random

    #for i in range(num_center):
    #    for j in range(i + 1, num_center):
    #       G.add_edge(centers[i], centers[j])
    edges = []
    # center_nodes = list(range(0, num_center))
    center_nodes = [f"M{num}" for num in range(0, num_center)]
    for i in range(num_center - 1):
        edges.append((center_nodes[i], center_nodes[i + 1]))
    end_nodes = [f"E{num}" for num in range(num_center, n)]
    for i in range(num_center, n):
        # 随机选择一个中心节点连接
        cidx = rng.randint(0, num_center - 1)
        eidx = i - num_center
        edges.append((center_nodes[cidx], end_nodes[eidx]))
    
    existing = {tuple(sorted(edge)) for edge in edges}
    candidates = [
        (center_nodes[i], center_nodes[j])
        for i in range(num_center)
        for j in range(i + 1, num_center)
        if tuple(sorted((center_nodes[i], center_nodes[j]))) not in existing
    ]
    rng.shuffle(candidates)
    edges.extend(candidates[:max(0, m - len(edges))])

    # 随机选择k对点对
    pairs = []
    # all_nodes = list(G.nodes)
    # end_nodes = [f"E{num}" for num in range(num_center, n)]
    # end_nodes = list(range(num_center, n))
    while len(pairs) < k:
        u, v = rng.sample(end_nodes, 2)  # 随机选择两个不同的节点
        pair = (u, v) if u < v else (v, u)
        if pair not in pairs:
            pairs.append(pair)

    return edges, pairs


def main():
    parser = argparse.ArgumentParser(description="Legacy synthetic graph generator")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-factor", type=float, default=0.02)
    parser.add_argument("--output-dir", default=".")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    ori_n = 500
    ori_k = 200
    scale_factor = args.scale_factor
    n = int(ori_n * scale_factor)
    m = int(n * 1.02)
    k = int(ori_k * scale_factor)
    c = int(n * 0.25)
    print(f'{n}, {m}, {k}, {c}, seed={args.seed}')
    new_edges, new_pairs = generate_connected_graph(n, m, k, c, rng)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / f'edges_{scale_factor}x.csv').open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'type', 'source', 'target', 'weight'])
        for edge in new_edges:
            writer.writerow(['', '', edge[0], edge[1], rng.randint(1, 50) * 10])

    with (output_dir / f'pairs_{scale_factor}x.csv').open(mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['id', 'type1', 'type2', 'source', 'target', 'weight'])
        for pair in new_pairs:
            writer.writerow(['', '', '', pair[0], pair[1], round(rng.randint(1, 9) / 1000, 3)])
    print("finish")


if __name__ == "__main__":
    main()


