from argparse import ArgumentParser
import os
import re

from transformers import AutoTokenizer

import yaml
from vllm import LLM, SamplingParams
import pandas as pd
from openai import OpenAI

from preprocess.run_amrparsing import convert_amrbart_v2_output


parser = ArgumentParser()
parser.add_argument("--dataset", type=str, required=True)
parser.add_argument("--prompting", type=str, required=True)
parser.add_argument("--llm", type=str, required=True)
args = parser.parse_args()

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
openai_client = OpenAI()


df = pd.read_json(f"examples/{args.dataset}.json", orient="records", lines=True)
df = df[~df['raw_subgraphs'].isna() ]

if 'sentence' in df.columns:
    df['sent'] = df['sentence']

# Load prompt templates
with open("prompts/base.yaml") as f:
    base_prompts = yaml.safe_load(f)
with open("prompts/amrcot.yaml") as f:
    amrcot_prompts = yaml.safe_load(f)
with open("prompts/subamrcot.yaml") as f:
    subamrcot_prompts = yaml.safe_load(f)
with open("prompts/predicate.yaml") as f:
    predicate_prompts = yaml.safe_load(f)
with open("prompts/entities.yaml") as f:
    entities_prompts = yaml.safe_load(f)
with open("prompts/realcot.yaml") as f:
    realcot_prompts = yaml.safe_load(f)


# prompting functions: row -> messages
def vanilla_prompting(row):
    messages = [
        {"role": "system", "content": base_prompts["system"]},
        {"role": "user", "content": base_prompts["user"].format(document=row["sent"])},
    ]
    return messages


def amrcot_prompting(row):
    messages = [
        {"role": "system", "content": amrcot_prompts["system"]},
        {
            "role": "user",
            "content": amrcot_prompts["user"].format(
                document=row["sent"], amr=row["amr"]
            ),
        },
    ]
    return messages


def realcot_prompting(row):
    messages = [
        {"role": "system", "content": realcot_prompts["system"]},
        {"role": "user", "content": realcot_prompts["user"]},
        {"role": "assistant", "content": realcot_prompts["assistant"]},
        {"role": "user", "content": realcot_prompts["user_next"].format(
            document=row["sent"], amr=row["amr"]
        )},
    ]
    return messages


def subamrcot_prompting(row):
    amr = ""
    for i, subgraph in enumerate(row["raw_subgraphs"]):
        penman = convert_amrbart_v2_output(subgraph)
        amr += f"## Subgraph {i+1}\n"
        amr += penman + "\n"

    messages = [
        {"role": "system", "content": subamrcot_prompts["system"]},
        {
            "role": "user",
            "content": subamrcot_prompts["user"].format(document=row["sent"], amr=amr),
        },
    ]
    return messages


def predicate_prompting(row):
    # extract predicates
    predicates = []
    predicates = re.findall(r"([a-z0-9-]+)-(\d+)\b", row["amr"])
    predicates = [x[0] for x in predicates]
    # print(predicates)

    messages = [
        {"role": "system", "content": predicate_prompts["system"]},
        {
            "role": "user",
            "content": predicate_prompts["user"].format(
                document=row["sent"], predicates=",".join(predicates)
            ),
        },
    ]
    return messages

def entities_prompting(row):
    entities = row['entities']
    if entities is None:
        entities = []
    messages = [
        {"role": "system", "content": entities_prompts["system"]},
        {
            "role": "user",
            "content": entities_prompts["user"].format(
                document=row["sent"], mentions=", ".join(entities)
            ),
        },
    ]
    return messages


# select prompting method
if args.prompting == "vanilla":
    prompting = vanilla_prompting
elif args.prompting == "amrcot":
    prompting = amrcot_prompting
elif args.prompting == "subamrcot":
    prompting = subamrcot_prompting
elif args.prompting == "predicate":
    prompting = predicate_prompting
elif args.prompting == "entities":
    prompting = entities_prompting
elif args.prompting == 'amrcoc':
    prompting = realcot_prompting
else:
    raise ValueError(f"Unknown prompting method: {args.prompting}")

all_messages = []
for idx, row in df.iterrows():
    messages = prompting(row)
    all_messages.append(messages)


def vllm_complete(all_messages):
    llm = LLM(model=model_name, enable_prefix_caching=True, tensor_parallel_size=2)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = []
    for messages in all_messages:
        chat_prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        if messages[-1]["role"] == "assistant":
            chat_prompt = chat_prompt.removesuffix("<|eot_id|>")
        # print(chat_prompt)
        prompts.append(chat_prompt)

    sampling_params = SamplingParams(max_tokens=4096, stop_token_ids=[128009])


    outputs = llm.generate(prompts, sampling_params=sampling_params)
    generated_texts = []
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        generated_text = generated_text.removeprefix(
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        )
        generated_text = generated_text.removeprefix(
            "Here is the rewritten paragraph:\n"
        )

        if args.prompting == "realcot" and "# Output" not in generated_text:
            generated_texts.append("<<FAILED>>")
            continue
        
        generated_text = generated_text.split('# Output')[-1]
        generated_text = generated_text.split('=>')[-1]

        generated_text = generated_text.strip()
        generated_texts.append(generated_text)
    return generated_texts


def openai_complete(all_messages):
    from tqdm import tqdm

    generations = []
    for messages in tqdm(all_messages):
        chat_completion = openai_client.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=messages,
            max_tokens=2048,
        )
        resp = chat_completion.choices[-1].message.content
        if args.prompting == "realcot" and "# Output" not in resp:
            generations.append("<<FAILED>>")
            continue
        generations.append(resp)
    return generations


if args.llm == "llama":
    generated_texts = vllm_complete(all_messages)
elif args.llm == "gpt":
    generated_texts = openai_complete(all_messages)
else:
    raise ValueError("llm must be llama or gpt")
df["simplification"] = generated_texts
df.to_json(
    f"llama-output/{args.dataset}-{args.llm}-{args.prompting}.json",
    orient="records",
    lines=True,
)
