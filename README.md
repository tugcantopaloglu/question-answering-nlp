# Evaluating Turkish BERT for Answer Retrieval

This research project evaluates whether question embeddings retrieve their paired answers from a fixed candidate pool. It uses the pretrained [dbmdz/bert-base-turkish-cased](https://huggingface.co/dbmdz/bert-base-turkish-cased) model and [merve/turkish_instructions](https://huggingface.co/datasets/merve/turkish_instructions). It performs inference and cosine ranking; it does not train a model or generate answers.

The original work by Tuğcan Topaloğlu is preserved in [ana_kod_tugcantopaloglu.ipynb](ana_kod_tugcantopaloglu.ipynb), the model-specific notebooks under `Modellere Özel Aynı Kod ama Tek Tek`, and the saved plots under `Model Grafikleri`. The Python script is the maintained command-line version. The notebooks retain their original Colab installation commands, CUDA assumptions, and recorded outputs.

## Setup

Use Python 3.10 or newer in a virtual environment. CPU inference is supported; a compatible PyTorch CUDA installation enables GPU inference. Downloading the dataset and pretrained model requires network access, local disk space, and substantially more time and memory than the fixture tests.

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then install:

```bash
python -m pip install -r requirements.txt
python ana_kod_tugcantopaloglu.py --help
```

For a CPU-only PyTorch wheel, install `torch` from the [official PyTorch CPU index](https://download.pytorch.org/whl/cpu) before installing the requirements:

```bash
python -m pip install "torch>=2.6,<3" --index-url https://download.pytorch.org/whl/cpu
```

Optional plots require `python -m pip install -r requirements-plot.txt`. Importing the script or running `--help` does not load the dataset or model and works without the ML dependencies.

## Run an evaluation

To select 1,000 paired rows deterministically, save them for reuse, and record the metrics:

```bash
python ana_kod_tugcantopaloglu.py --sample-size 1000 --seed 42 --batch-size 16 --save-csv selected_questions_answers.csv --output-json retrieval-results.json
```

The default device is `auto`, which uses CUDA when available and otherwise CPU. Add `--device cpu` to force CPU inference. Reduce `--batch-size` if memory is limited. A smaller `--sample-size`, such as 10, is suitable for a smoke run, but its scores are not comparable to a 1,000-candidate experiment.

To reuse every row of an existing UTF-8 CSV with `Question` and `Answer` columns:

```bash
python ana_kod_tugcantopaloglu.py --csv selected_questions_answers.csv --device cpu --batch-size 16 --output-json retrieval-results.json
```

`--sample-size` and `--seed` control dataset sampling; CSV mode preserves its rows and order. `--seed` also seeds PyTorch. The default model is pinned to `b6e1de16c983e0f2c70664591ea3f22810072608` and the default dataset to `ed3381a23d9328b02054456d8ca5bdc157f617d5`. Remote overrides require `--model-revision` or `--dataset-revision` with a full 40-character Hub commit hash; local directories do not require a revision. The result JSON records the model revision when available, source information, ordered-pair SHA-256 digest, device, effective token limit, pooling method, candidate count, and rates in the range 0 to 1. Dependencies are compatible version ranges rather than an experiment lockfile; save the environment versions with your results. A fixed seed does not guarantee identical floating-point results across devices or library versions.

`--local-files-only` applies to the model and tokenizer. Use it with `--csv` for a run that does not need a dataset download; the model must already be cached or supplied as a local directory with `--model`. No custom Hub Python code is enabled. A model override must support the BERT-style leading-token representation and positional limit used by this script. Other embedding models may require different pooling, prompts, or normalization, so simply changing `--model` is not a valid comparison for every architecture.

To save a t-SNE plot, install the plotting requirements and add `--plot retrieval-tsne.png`. Plotting reuses the evaluated embeddings and adds computation. It adapts perplexity to small samples, and it is a qualitative visualization rather than an evaluation metric.

## Data and evaluation contract

The source notebook records dataset columns `talimat`, ` giriş`, and ` çıktı`, including the leading spaces. The script strips column-name whitespace and combines instruction and optional input into each question. It rejects ambiguous columns, missing or blank questions/answers, and invalid sample sizes instead of silently dropping rows. CSV loading preserves Turkish text, quoted newlines, and literal strings such as `NA`.

The tokenizer uses padding and truncation, with a default maximum of 512 tokens bounded by the tokenizer and model positional limits. Embeddings are the last hidden state's leading `[CLS]` vector. Inference disables dropout and gradient recording, encodes each question and answer once in batches, and returns embeddings to CPU before cosine ranking. Ranking itself operates in question batches.

The correct answer is the answer at the same row index as the question. Top-1 counts exact first-place retrieval; top-k counts inclusion among the first k candidates, capped at the candidate-pool size. Ties use ascending candidate row order. Duplicate answer strings remain separate candidates, so an equivalent answer at another index does not count as correct. These are paired-row retrieval rates, not human judgments of answer correctness or general semantic similarity. Empty, nonfinite, zero-norm, or misaligned embeddings are rejected.

## Historical results

The saved main notebook reports **top-1: 3.50%** and **top-5: 4.90%** for `dbmdz/bert-base-turkish-cased` on its randomly selected 1,000 pairs. These are historical outputs, not a newly reproduced benchmark. The exact sampled CSV, sampling seed, and pinned model/dataset revisions are not committed, and the original code leaves dropout enabled. The maintained script corrects that behavior; its results must be reported as a separate run with its own saved sample and environment.

The repository also contains notebooks or plots for `HIT-TMG/KaLM-embedding-multilingual-mini-v1`, `pingkeest/learning2_model`, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`, `Alibaba-NLP/gte-multilingual-base`, and `Alibaba-NLP/gte-large-en-v1.5`. Those artifacts preserve the original experiments and do not establish that this script implements each model's recommended embedding pipeline. Some notebooks also explore answer-to-question retrieval.

## Local validation

```bash
python -m pip install -r requirements-dev.txt -r requirements-plot.txt
python -m unittest discover -s tests -v
python -m bandit ana_kod_tugcantopaloglu.py --skip B311
python -m pip_audit --progress-spinner off
```

Tests create a tiny randomly initialized BERT and Turkish tokenizer entirely in a temporary local directory. They check CPU loading, token limits, disabled dropout, batched inference, paired-row cosine ranking, invalid data, Unicode CSV round trips, CLI result metadata, and small t-SNE plots. These fixtures test execution and evaluation logic; their scores say nothing about the pretrained model's accuracy. CI sets the Hugging Face offline flags and runs the same fixtures and security checks without downloading weights or datasets.

Bandit's B311 rule is excluded explicitly because the seeded pseudo-random generator selects reproducible research rows and is not used for security tokens or cryptography. Other Bandit checks remain enabled, including checks for unsafe Hub downloads.

## License

The project is distributed under the [MIT license](LICENSE). The referenced datasets and pretrained models retain their own terms and attribution.
