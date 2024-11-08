"""
Our AMRSSS pipeline.
Maybe: input entire document, run senter as the first step
Input: raw sentence
Step 1: apply vanilla AMRBART parser; output: AMR graph
Step 2: apply graph splitting algorithm; output: multiple AMR graphs
Step 3: Mask entities
Step 4: Apply finetuned AMR2Text model; output: multiple sentences
Step 5: Unmask entities
"""

import argparse
import pathlib
from tempfile import TemporaryDirectory
import os
import logging
import json
import subprocess
from contextlib import contextmanager
import pickle
from functools import wraps

import pandas as pd

from preprocess.run_amrparsing import parse_sentences
from amr_format import parse_amrbart_output
from amr_operations import AMRGraph


# a decorator that caches function output to pickle; takes tmp path as a parameter
def cached(tmp_dir):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            cache_path = os.path.join(tmp_dir, f"{func.__name__}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    return pickle.load(f)
            else:
                result = func(*args, **kwargs)
                with open(cache_path, "wb") as f:
                    pickle.dump(result, f)
                return result

        return wrapper

    return decorator


@contextmanager
def open_tempdir(path=None):
    if path is None:
        dir = TemporaryDirectory()
        try:
            yield dir.name
        finally:
            dir.cleanup()
    else:
        try:
            yield path
        finally:
            pass


def resolve(path):
    return str((pathlib.Path(__file__).parent / pathlib.Path(path)).resolve())


def run_amr_to_text(amrgraphs, amr_model, linearized=False):
    with TemporaryDirectory() as amr_input_dir, TemporaryDirectory() as amr_output_dir:
        logging.info("amr ourput dir: %s", amr_output_dir)
        with open(f"{amr_input_dir}/data4generation.jsonl", "w") as amr_input_file:
            for node in amrgraphs:
                if not linearized:
                    amr = node.to_spring(delim=" ", lit_begin='"', lit_end='"')
                else:
                    amr = node
                amr_input_file.write(
                    json.dumps(
                        {
                            "sent": "",
                            "amr": amr,
                        }
                    )
                    + "\n"
                )
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = "20"
        # run amrbart
        amr_script = resolve("amrbart/amrbart-v2/fine-tune/main.py")
        # fmt: off
        cmd_args = [
            "--data_dir", amr_input_dir,
            "--task", "amr2text",
            "--test_file", f"{amr_input_dir}/data4generation.jsonl",
            "--output_dir", amr_output_dir,
            "--cache_dir", "/tmp",
            "--data_cache_dir", "/tmp",
            "--overwrite_output_dir",
            "--overwrite_cache", "True",
            "--model_name_or_path", amr_model,
            "--unified_input", "True",
            "--per_device_eval_batch_size", "8",
            "--max_source_length", "1024",
            "--max_target_length", "400",
            "--val_max_target_length", "400",
            "--generation_max_length", "400",
            "--generation_num_beams", "5",
            "--predict_with_generate",
            "--smart_init", "False",
            "--use_fast_tokenizer", "False",
            "--logging_dir", f"{amr_output_dir}/logs",
            "--seed", "42",
            "--dataloader_num_workers", "8",
            "--eval_dataloader_num_workers", "2",
            "--include_inputs_for_metrics",
            "--do_predict",
            "--ddp_find_unused_parameters", "False",
            "--report_to", "tensorboard",
            "--dataloader_pin_memory", "True",
        ]
        # fmt: on
        cmd_args += ["--fp16_backend", "auto", "--fp16"]
        subprocess.run(["python3", amr_script] + cmd_args, env=env, check=True)

        with open(f"{amr_output_dir}/generated_predictions.txt") as f:
            return f.readlines()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AMRSSS pipeline")
    parser.add_argument("input", type=pathlib.Path, help="Input file")
    parser.add_argument(
        "--input_col", type=str, default="sentence", help="Input column"
    )
    parser.add_argument("--tmp_dir", default="/tmp", type=pathlib.Path, help="Temporary directory")
    parser.add_argument(
        "--debug", default=False, action="store_true", help="Write debug output"
    )
    parser.add_argument(
        "--clear_cache", default=False, action="store_true", help="Clear cache"
    )
    parser.add_argument(
        "--invert_edges",
        default=False,
        action="store_true",
        help="Invert edges ending with -of",
    )
    parser.add_argument(
        "--amr_model",
        type=str,
        default="./amrbart/realization-model",
        help="AMR2Text model",
    )
    parser.add_argument("output", type=pathlib.Path, help="Output file")
    args = parser.parse_args()

    if args.clear_cache:
        os.system(f"rm -rf {args.tmp_dir}/_*.pkl")

    with open_tempdir(args.tmp_dir) as temp_dir:
        print("Temp dir:", temp_dir)
        # Step 0: read input file
        if args.input.suffix == ".csv":
            df = pd.read_csv(args.input)
            input_sentences = df[args.input_col].unique().tolist()
        elif args.input.suffix == ".json":
            df = pd.read_json(args.input, lines=True)
            input_sentences = df[args.input_col].unique().tolist()
        elif args.input.suffix == ".txt":
            with open(args.input) as f:
                input_sentences = f.readlines()
            df = pd.DataFrame({args.input_col: input_sentences})
        else:
            raise ValueError("Input file must be .csv, .json, or .txt")

        # Step 1: apply vanilla AMRBART parser; output: AMR graph
        print("Parsing sentences...")

        @cached(temp_dir)
        def _parse_sentences(sentences):
            amr_penman = parse_sentences(sentences)
            return amr_penman

        amr_penman = _parse_sentences(input_sentences)
        # with open(os.path.join(temp_dir, "amr_penman.txt"), "w") as f:
        #    f.write(amr_penman)

        amrs = []
        with open(os.path.join(temp_dir, "amr_penman.txt"), "w") as f:
            # Make sure it's parseable by parse_amrbart_output
            # In the meantime, add intent
            for sent_idx, (sent, amr_node, amr_output) in enumerate(
                parse_amrbart_output(amr_penman)
            ):
                f.write(f"# ::sent {sent}\n")
                amr = amr_node.to_penman()
                amrs.append(amr)
                f.write(amr + "\n\n")

        # Step 2: apply graph splitting algorithm; output: multiple AMR graphs
        print("Extracting subgraphs...")
        dropped_sents_indices = []

        def _extract_subgraphs(amr_penman):
            num_subgraphs_per_sent = []
            all_subgraphs = []
            for sent_idx, (sent, amr_node, amr_output) in enumerate(
                parse_amrbart_output(amr_penman)
            ):
                if amr_node is None:
                    print(f"sent idx missed: {sent_idx}")
                    dropped_sents_indices.append(sent_idx)
                    num_subgraphs_per_sent.append(0)
                    continue
                graph = AMRGraph(amr_node, invert_edges=args.invert_edges)
                subgraphs = graph.extract_subgraphs()
                num_subgraphs_per_sent.append(len(subgraphs))
                for i, subgraph in enumerate(subgraphs):
                    all_subgraphs.append((sent, amr_node, i, subgraph, amr_output))
            return num_subgraphs_per_sent, all_subgraphs

        num_subgraphs_per_sent, all_subgraphs = _extract_subgraphs(amr_penman)
        assert any(x > 0 for x in num_subgraphs_per_sent)
        with open(os.path.join(temp_dir, "subgraphs.txt"), "w") as f:
            for sent, _, i, subgraph, _ in all_subgraphs:
                f.write(f" ::sent {sent}\n")
                f.write(f" ::subgraph {i}\n")
                f.write(subgraph.to_penman() + "\n\n")

        # Step 3: Mask entities
        print("Masking entities...")
        all_mask_entity_pairs = []
        for _, _, _, subgraph, _ in all_subgraphs:
            mask_entity_pairs = subgraph.mask_entities()
            all_mask_entity_pairs.append(mask_entity_pairs)
        with open(os.path.join(temp_dir, "masked_subgraphs.txt"), "w") as f:
            for sent, _, i, subgraph, _ in all_subgraphs:
                f.write(f"# ::sent {sent}\n")
                f.write(f"# ::subgraph {i}\n")
                f.write(subgraph.to_penman() + "\n\n")
        with open(os.path.join(temp_dir, "masked_entity_pairs.txt"), "w") as f:
            for pairs in all_mask_entity_pairs:
                f.write(str(pairs) + "\n")

        # Step 4: Apply finetuned AMR2Text model; output: multiple sentences
        print("Running AMR2Text model...")

        @cached(temp_dir)
        def _run_amr_to_text(all_subgraphs):
            amr2text_model = args.amr_model
            output_sentences = run_amr_to_text(
                [subgraph for _, _, _, subgraph, _ in all_subgraphs],
                amr2text_model,
            )
            return output_sentences

        output_sentences = _run_amr_to_text(all_subgraphs)
        output_sentences = [
            max(x.split("|"), key=len) for x in output_sentences
        ]  # remove wiki markup if any
        with open(os.path.join(temp_dir, "amr2text_output.txt"), "w") as f:
            f.writelines(output_sentences)

        # Step 5: Unmask entities
        print("Unmasking entities...")
        unmasked_sentences = []
        for pairs, sent in zip(all_mask_entity_pairs, output_sentences):
            for mask, entity in sorted(
                pairs, key=lambda x: len(x[0]), reverse=True
            ):
                sent = sent.replace(mask, entity)
            unmasked_sentences.append(sent)

        # Squeeze
        simplified_sent_offset = 0
        output_sentences = []
        for sent_idx, num_subgraphs in enumerate(num_subgraphs_per_sent):
            # restore missing sentences
            if num_subgraphs == 0:
                output_sentences.append(input_sentences[sent_idx])
                continue
            sents = unmasked_sentences[
                simplified_sent_offset : simplified_sent_offset + num_subgraphs
            ]
            sents = [x.strip() for x in sents]
            output_sentences.append("  ".join(sents))
            simplified_sent_offset += num_subgraphs

        # Write output
        output_df = pd.DataFrame(
            {"sentence": input_sentences, "simplification": output_sentences, "amr": amrs}
        )
        output_df["sentence"] = output_df["sentence"].apply(lambda x: x.strip())
        output_df["simplification"] = output_df["simplification"].apply(
            lambda x: x.strip()
        )
        for col in df.columns:
            if col not in output_df.columns:
                output_df[col] = df[col]

        output_df.to_csv(args.output, index=False)
