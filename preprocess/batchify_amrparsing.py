import os
import json

chunks = 25

with open("/data/local/amrs3/simplewiki-sentences.txt") as f:
    lines = f.readlines()
    chunksize = len(lines) // chunks 
    for n in range(chunks):
        os.makedirs(f"/data/local/amrs3/chunked-input/simplewiki-{n:02d}", exist_ok=True)
        with open(f"/data/local/amrs3/chunked-input/simplewiki-{n:02d}/test.jsonl", "w") as f:
            for line in lines[n * chunksize : (n + 1) * chunksize]:
                f.write(json.dumps({"src": line.strip(), "tgt": ""}) + "\n")

        with open(f"/data/local/amrs3/chunked-input/simplewiki-{n:02d}/train.jsonl", "w") as f:
            f.write('{"src": "", "tgt": ""}\n')
        with open(f"/data/local/amrs3/chunked-input/simplewiki-{n:02d}/val.jsonl", "w") as f:
            f.write('{"src": "", "tgt": ""}\n')
