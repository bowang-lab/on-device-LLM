#!/usr/bin/env python3
import json
from pprint import pprint

path = "train_cot_long.jsonl"
n_show = 3  # change to how many you want to see

with open(path, "r") as f:
    for i, line in enumerate(f):
        if i >= n_show:
            break
        print(f"\n=== Example {i} ===")
        ex = json.loads(line)
        pprint(ex, width=120)
