import json
from pathlib import Path

from tqdm import tqdm

DATA_DIR = Path("/data/local/amrs3/")

for name in ("train", "val", "test"):
    with open(DATA_DIR / "finetune-masked-2" / f"{name}.jsonl") as input_f,\
            open(DATA_DIR / "finetune-unmasked" / f"{name}.jsonl", "w") as output_f:
        for line in tqdm(input_f):
            record = json.loads(line)
            items = list(record["ents"].items())
            items.sort(key=lambda x: len(x[0]), reverse=True)
            for key, value in items:
                record["sent"] = record["sent"].replace(key, value)
                record["amr"] = record["amr"].replace(key, value)
            output_f.write(json.dumps(record) + "\n")
