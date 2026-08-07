import csv
import time
from datetime import datetime
from typing import Dict, List, Set, Tuple, Optional


def read_csv_to_pairs(file_path):
    pairs_to_weight = {}

    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for tokens in reader:
            if len(tokens) >= 5:
                node1_name = tokens[3]
                node2_name = tokens[4]
                weight = float(tokens[5])
                pair = (node1_name, node2_name)
                pairs_to_weight[pair] = weight

    return pairs_to_weight

pairs_to_weight = read_csv_to_pairs("./pairs-4-246.csv")
total_weight = 0
with open("./prisol-246.csv", 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip header
        for tokens in reader:
            if len(tokens) >= 5:
                node1_name = tokens[0]
                node2_name = tokens[1]
                length = float(tokens[2])
                pair = (node1_name, node2_name)
                if pair not in pairs_to_weight:
                    print('pair error')
                print(pairs_to_weight[pair])
                print(length)
                total_weight += length * pairs_to_weight[pair]
print(total_weight)
