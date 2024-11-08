import json

from tqdm import tqdm
from datasets import load_dataset
import spacy
import pysbd

seg = pysbd.Segmenter(language="en", clean=False)
nlp = spacy.load("en_core_web_md")

def process_text(inputs):
    line_idx, text = inputs

    sents = []
    skipped = []
    for sent_idx, sent in enumerate(seg.segment(text)):
        alpha_cnt = 0
        for _, token in enumerate(nlp(sent)):
            if token.text.isalpha():
                alpha_cnt += 1
            if alpha_cnt == 5:
                break
        if alpha_cnt < 5:
            skipped.append((line_idx, sent_idx, sent))
            continue
        sents.append((line_idx, sent_idx, sent))
    return sents, skipped


if __name__ == '__main__':
    import multiprocessing as mp
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    
    with mp.Pool(mp.cpu_count() - 1) as pool, \
        open("../data/wikitext-103-train-sentences-pysbd.jsonl", "w") as sents_f, \
        open("../data/wikitext-103-train-skipped-pysbd.jsonl", "w") as skipped_f:
        results = pool.imap_unordered(process_text, enumerate(dataset["text"]), chunksize=1000)
        for sents, skipped in tqdm(results, total=len(dataset)):
            for s in sents:
                sents_f.write(json.dumps({"line_idx": s[0], "sent_idx": s[1], "text": s[2]}) + "\n")
            for s in skipped:
                skipped_f.write(json.dumps({"line_idx": s[0], "sent_idx": s[1], "text": s[2]}) + "\n")
