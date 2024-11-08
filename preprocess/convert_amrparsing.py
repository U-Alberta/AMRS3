# convert the output of senter to input of amrparser
import json

with open("../data/wikitext-103-train-sentences-pysbd.jsonl", "r") as f:
    lines = [json.loads(line) for line in f]

with open("../data/wikidata-103-train-amr/test.jsonl", "w") as f:
    for line in lines:
        f.write(
            json.dumps({"src": line["text"], "tgt": ""}) + "\n"
        )
with open("../data/wikidata-103-train-amr/train.jsonl", "w") as f:
    f.write(json.dumps({"src": "", "tgt": ""}) + "\n")
with open("../data/wikidata-103-train-amr/val.jsonl", "w") as f:
    f.write(json.dumps({"src": "", "tgt": ""}) + "\n")