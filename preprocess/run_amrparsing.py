import importlib
from pathlib import Path
import sys
import os
import subprocess
from tempfile import TemporaryDirectory
import logging
import json
import re


def resolve(path):
    return str((Path(__file__).parent / Path(path)).resolve())


def convert_amrbart_v2_output(v2_output):
    # v2 output -> v1 output
    output = v2_output
    # try to fix mismatched brackets
    if output.count("(") != output.count(")"):
        tokens = []
        for line in output.splitlines():
            tokens.extend(line.strip().split())
            tokens.append('\n')
        tokens.pop()
        open_paren = 0
        tracking = True
        for i, token in enumerate(tokens):
            if token == '\n':
                continue
            if token == '<lit>':
                tracking = False
            elif token == '</lit>':
                tracking = True
            if not tracking:
                continue
            if token == '(':
                open_paren += 1
            elif token == ')':
                if open_paren == 0:
                    tokens[i] = ''
                else:
                    open_paren -= 1
        output = " ".join(tokens)
    # convert <pointer:N> to zN
    output = re.sub(r"<pointer:(\d+)>", r"z\1 /", output, flags=re.MULTILINE)
    # convert zN / ) to zN) (if zN is a reference and is the last argument)
    output = re.sub(r"z(\d+) / \)", r"z\1 )", output, flags=re.MULTILINE)
    # convert zN / : to zN : (if zN is a reference and is not the last argument)
    output = re.sub(r"z(\d+) / :", r"z\1 :", output, flags=re.MULTILINE)
    # convert <lit> and </lit> to "
    output = re.sub(r"<lit> ", r'"', output, flags=re.MULTILINE)
    output = re.sub(r" </lit>", r'"', output, flags=re.MULTILINE)

    return output



def parse_sentences(sentences, return_amrbart_format=False):
    # Text to AMR
    # sentences = [str]
    # output -> AMR graphs, in PENMAN format, in the same order as the input sentences
    # separated by newlines
    if len(sentences) == 0:
        return ""
    with TemporaryDirectory() as amr_input_dir, TemporaryDirectory() as amr_output_dir:
        logging.info("amr ourput dir: %s", amr_output_dir)
        with open(f"{amr_input_dir}/test.jsonl", "w") as amr_input_file:
            for sent in sentences:
                amr_input_file.write(json.dumps({"sent": sent, "amr": ""}) + "\n")
        call_amrbart_amrparsing(amr_input_dir, amr_output_dir)

        with open(f"{amr_output_dir}/generated_predictions.txt") as f:
            raw_output = f.readlines()
        assert len(raw_output) == len(sentences)

        if not return_amrbart_format:
            output_lines = []
            for sent, amr in zip(sentences, raw_output):
                sent = sent.replace("\n", " ")
                amr = amr.replace("</AMR>", "")
                amr = convert_amrbart_v2_output(amr)
                output_lines.append(f"# ::snt {sent}")
                output_lines.append(amr)
                output_lines.append("")
            return "\n".join(output_lines)
        else:
            amrbart_output = []
            for sent, amr in zip(sentences, raw_output):
                sent = sent.replace("\n", " ")
                amr = amr.replace("</AMR>", "")
                amrbart_output.append({"sent": sent, "amr": amr})
            return amrbart_output


def call_amrbart_amrparsing(amr_input_dir, amr_output_dir):
    amr_model = resolve("../amrbart/parsing-model")
    amr_script = resolve("../amrbart/amrbart-v2/fine-tune/main.py")

    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "16"
    # run amrbart
    cmd_args = {
        "data_dir": amr_input_dir,
        "task": "text2amr",
        "test_file": amr_input_dir + "/test.jsonl",
        "output_dir": amr_output_dir,
        "data_cache_dir": "/tmp",
        "overwrite_cache": True,
        "model_name_or_path": amr_model,
        "overwrite_output_dir": "",
        "unified_input": True,
        "per_device_eval_batch_size": 16,
        "max_source_length": 300,
        "max_target_length": 768,
        "val_max_target_length": 768,
        "generation_max_length": 768,
        "generation_num_beams": 5,
        "predict_with_generate": "",
        "smart_init": False,
        "use_fast_tokenizer": False,
        "logging_dir": "/tmp",
        "seed": 42,
        "dataloader_num_workers": 16,
        "eval_dataloader_num_workers": 16,
        "do_predict": "",
        "include_inputs_for_metrics": "",
        "ddp_find_unused_parameters": False,
        "report_to": "tensorboard",
        "dataloader_pin_memory": True,
    }

    cmd_args["fp16"] = ""
    cmd_args["fp16_backend"] = "auto"
    cmd_args["fp16_full_eval"] = ""

    cmd_args_array = []
    for k, v in cmd_args.items():
        cmd_args_array.append(f"--{k}")
        if v != "":
            cmd_args_array.append(str(v))
    cmd_args = cmd_args_array

    print(cmd_args)
    subprocess.run(
        ["python3", amr_script]
        + cmd_args,
        env=env,
    )