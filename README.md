# AMRSSS & AMRCoC
Peiran Yao, Kostyantyn Guzhva, and Denilson Barbosa. 2024. [Semantic Graphs for Syntactic Simplification: A Revisit from the Age of LLM](https://aclanthology.org/2024.textgraphs-1.8/). In Proceedings of TextGraphs-17: Graph-based Methods for Natural Language Processing, pages 105–115, Bangkok, Thailand. Association for Computational Linguistics.

Unfortunately, we are still working on licensing to release the exact Orlando dataset used in the paper - we will update this repository after EMNLP 2024.

## Setting up environment
```bash
conda env create --name amrbart --file requirements.yml
conda activate amrbart
```

Unfortunately, you might still need to manually downgrade numpy and pandas:
```bash
pip install numpy=1.26.4
pip install "pandas<2.2"
```

## Downloading the models
```bash
git lfs install
git clone https://huggingface.co/xfbai/AMRBART-large-finetuned-AMR3.0-AMRParsing-v2 amrbart/parsing-model
git clone https://huggingface.co/peiran-yao/AMRBART-AMR2Text-SimpleWiki-EntityMasking amrbart/realization-model
```

## Running the AMRBART-based pipeline (AMRSSS)
```bash
python3 pipeline-bart.py example.csv example-output.csv
```

## Running the LLM+AMR pipelines (AMRCoC)

Unfortunately, AMRBART's dependencies are too old for the LLM+AMR pipelines. You will need to create a new environment for this:
```bash
conda env create --name amr-coc --file amr-coc/requirements.yml
conda activate amr-coc
pip install vllm openai
```

Then, you can run the pipeline:

Change line 25 in `pipeline-llm.py` to the path of AMRSSS output file (or anything that contains `amr` column).
```bash
python3 pipeline-llm.py --prompting "vanilla"/"amrcot"/"amrcoc" --llm gpt
```

## Citation
```bibtex
@inproceedings{yao-etal-2024-semantic,
    title = "Semantic Graphs for Syntactic Simplification: A Revisit from the Age of {LLM}",
    author = "Yao, Peiran  and
      Guzhva, Kostyantyn  and
      Barbosa, Denilson",
    booktitle = "Proceedings of TextGraphs-17: Graph-based Methods for Natural Language Processing",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.textgraphs-1.8",
    pages = "105--115"
}