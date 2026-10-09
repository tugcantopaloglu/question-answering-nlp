"""Turkish BERT answer retrieval, adapted from Tuğcan Topaloğlu's notebook."""

import argparse
import csv
import hashlib
import json
import random
import re
from pathlib import Path


MODEL_NAME = "dbmdz/bert-base-turkish-cased"
DATASET_NAME = "merve/turkish_instructions"
MODEL_REVISION = "b6e1de16c983e0f2c70664591ea3f22810072608"
DATASET_REVISION = "ed3381a23d9328b02054456d8ca5bdc157f617d5"


def _asset_revision(name, revision, default_name, default_revision):
    if Path(name).is_dir():
        return None
    if revision is None and name == default_name:
        revision = default_revision
    if not isinstance(revision, str) or re.fullmatch(r"[0-9a-fA-F]{40}", revision) is None:
        raise ValueError("Remote models and datasets require a full 40-character commit revision.")
    return revision


def _validate_pairs(questions, answers):
    if not questions or len(questions) != len(answers):
        raise ValueError("Questions and answers must be nonempty and have equal lengths.")
    if any(not isinstance(text, str) or not text.strip() for text in [*questions, *answers]):
        raise ValueError("Every question and answer must be a nonempty string.")


def select_question_answer_pairs(records, sample_size=1000, seed=42):
    questions, answers = [], []
    for index, record in enumerate(records):
        fields = {key.strip(): value for key, value in record.items()}
        if len(fields) != len(record):
            raise ValueError(f"Row {index} has conflicting column names after stripping spaces.")
        instruction = fields.get("talimat")
        input_text = fields.get("giriş", "")
        response = fields.get("çıktı")
        if input_text is None:
            input_text = ""
        if not all(isinstance(value, str) for value in (instruction, input_text, response)):
            raise ValueError(f"Row {index} requires string talimat, giriş and çıktı fields.")
        questions.append(f"{instruction} {input_text}".strip())
        answers.append(response.strip())
    _validate_pairs(questions, answers)
    if sample_size <= 0 or sample_size > len(questions):
        raise ValueError(f"Sample size must be between 1 and {len(questions)}.")
    indices = random.Random(seed).sample(range(len(questions)), sample_size)
    return [questions[index] for index in indices], [answers[index] for index in indices]


def load_question_answer_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames is None or not {"Question", "Answer"}.issubset(reader.fieldnames):
            raise ValueError("CSV must contain Question and Answer columns.")
        if len(set(reader.fieldnames)) != len(reader.fieldnames):
            raise ValueError("CSV column names must be unique.")
        rows = list(reader)
    if any(None in row for row in rows):
        raise ValueError("CSV rows must not contain extra fields.")
    questions = [row["Question"] for row in rows]
    answers = [row["Answer"] for row in rows]
    _validate_pairs(questions, answers)
    return questions, answers


def save_question_answer_csv(path, questions, answers):
    _validate_pairs(questions, answers)
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["Question", "Answer"])
        writer.writerows(zip(questions, answers))


def get_model_and_tokenizer(model_name, device="auto", revision=None, local_files_only=False):
    revision = _asset_revision(model_name, revision, MODEL_NAME, MODEL_REVISION)
    import torch
    from transformers import AutoModel, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable; use --device cpu.")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision, local_files_only=local_files_only, trust_remote_code=False)
    model = AutoModel.from_pretrained(model_name, revision=revision, local_files_only=local_files_only, trust_remote_code=False).to(device)
    model.eval()
    return tokenizer, model


def get_embeddings(texts, tokenizer, model, max_length=512):
    return get_embeddings_in_batches(texts, tokenizer, model, batch_size=len(texts), max_length=max_length)


def get_embeddings_in_batches(texts, tokenizer, model, batch_size=64, max_length=512):
    import torch

    if batch_size <= 0 or max_length <= 0:
        raise ValueError("Batch size and maximum token length must be positive.")
    if not texts or any(not isinstance(text, str) or not text.strip() for text in texts):
        raise ValueError("Texts must be a nonempty sequence of nonempty strings.")
    limits = [max_length, tokenizer.model_max_length, model.config.max_position_embeddings]
    token_limit = min(limits)
    model.eval()
    embeddings = []
    with torch.inference_mode():
        for start in range(0, len(texts), batch_size):
            inputs = tokenizer(
                list(texts[start:start + batch_size]), return_tensors="pt",
                padding=True, truncation=True, max_length=token_limit,
            )
            inputs = {name: value.to(model.device) for name, value in inputs.items()}
            outputs = model(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].cpu())
    return torch.cat(embeddings)


def _normalized_embeddings(embeddings):
    import numpy as np

    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 2 or not all(values.shape) or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a nonempty, finite two-dimensional array.")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms == 0).any():
        raise ValueError("Embeddings must have finite, nonzero norms.")
    return values / norms


def calculate_similarity(embedding1, embedding2):
    import numpy as np

    left, right = np.asarray(embedding1), np.asarray(embedding2)
    if left.ndim != 1 or right.ndim != 1 or left.shape != right.shape:
        raise ValueError("Similarity requires two vectors of equal length.")
    vectors = _normalized_embeddings([left, right])
    return float(vectors[0] @ vectors[1])


def evaluate_embeddings(question_embeddings, answer_embeddings, k=5, batch_size=64):
    import numpy as np

    if k <= 0 or batch_size <= 0:
        raise ValueError("Top-k and batch size must be positive.")
    questions = _normalized_embeddings(question_embeddings)
    answers = _normalized_embeddings(answer_embeddings)
    if questions.shape != answers.shape:
        raise ValueError("Question and answer embeddings must have matching shapes and row order.")
    candidate_count = len(answers)
    top1_count, topk_count = 0, 0
    for start in range(0, len(questions), batch_size):
        scores = questions[start:start + batch_size] @ answers.T
        ranked = np.argsort(-scores, axis=1, kind="stable")[:, :min(k, candidate_count)]
        expected = np.arange(start, start + len(ranked))
        top1_count += int((ranked[:, 0] == expected).sum())
        topk_count += int((ranked == expected[:, None]).any(axis=1).sum())
    return top1_count / candidate_count, topk_count / candidate_count


def calculate_top_k_success(questions, answers, tokenizer, model, k=5, batch_size=64, max_length=512):
    _validate_pairs(questions, answers)
    question_embeddings = get_embeddings_in_batches(questions, tokenizer, model, batch_size, max_length)
    answer_embeddings = get_embeddings_in_batches(answers, tokenizer, model, batch_size, max_length)
    return evaluate_embeddings(question_embeddings, answer_embeddings, k=k, batch_size=batch_size)


def plot_tsne(embeddings, labels, title, output_path=None):
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE

    values = np.asarray(embeddings)
    _normalized_embeddings(values)
    if len(values) < 2 or values.shape[1] < 2 or len(values) != len(labels):
        raise ValueError("t-SNE requires at least two vectors, two dimensions, and one label per vector.")
    reduced = TSNE(n_components=2, random_state=42, perplexity=min(30, len(values) - 1)).fit_transform(values)
    colors = ["blue" if label == "question" else "red" for label in labels]
    figure, axis = plt.subplots(figsize=(10, 8))
    axis.scatter(reduced[:, 0], reduced[:, 1], c=colors, alpha=0.5)
    axis.set(title=title, xlabel="t-SNE Dim 1", ylabel="t-SNE Dim 2")
    if output_path is None:
        plt.show()
    else:
        figure.savefig(output_path, bbox_inches="tight")
    plt.close(figure)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate paired-row retrieval using Turkish BERT CLS embeddings.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--csv", type=Path, help="Reuse all paired rows from a Question,Answer CSV.")
    source.add_argument("--dataset", default=DATASET_NAME, help="Hugging Face dataset to sample.")
    parser.add_argument("--split", default="train")
    parser.add_argument("--dataset-revision", help="Dataset revision; use a commit hash for reproducibility.")
    parser.add_argument("--sample-size", type=int, default=1000, help="Number of dataset rows to sample; ignored with --csv.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-csv", type=Path, help="Save the exact evaluated question and answer pairs.")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--model-revision", help="Model revision; use a commit hash for reproducibility.")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--local-files-only", action="store_true", help="Load the model and tokenizer only from local cache.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--plot", type=Path, help="Save an optional t-SNE plot; adds computation.")
    args = parser.parse_args(argv)
    sizes = [args.batch_size, args.max_length, args.top_k]
    if args.csv is None:
        sizes.append(args.sample_size)
    if min(sizes) <= 0:
        parser.error("Sample size, batch size, maximum length and top-k must be positive.")
    model_revision = _asset_revision(args.model, args.model_revision, MODEL_NAME, MODEL_REVISION)
    if args.csv:
        questions, answers = load_question_answer_csv(args.csv)
        data_source = {"csv": str(args.csv)}
    else:
        dataset_revision = _asset_revision(args.dataset, args.dataset_revision, DATASET_NAME, DATASET_REVISION)
        from datasets import load_dataset

        dataset = load_dataset(args.dataset, split=args.split, revision=dataset_revision)
        questions, answers = select_question_answer_pairs(dataset, args.sample_size, args.seed)
        data_source = {"dataset": args.dataset, "split": args.split, "revision": dataset_revision}
    if args.save_csv:
        save_question_answer_csv(args.save_csv, questions, answers)
    import torch

    torch.manual_seed(args.seed)
    tokenizer, model = get_model_and_tokenizer(args.model, args.device, model_revision, args.local_files_only)
    question_embeddings = get_embeddings_in_batches(questions, tokenizer, model, args.batch_size, args.max_length)
    answer_embeddings = get_embeddings_in_batches(answers, tokenizer, model, args.batch_size, args.max_length)
    top1, topk = evaluate_embeddings(question_embeddings, answer_embeddings, args.top_k, args.batch_size)
    pairs_json = json.dumps(list(zip(questions, answers)), ensure_ascii=False, separators=(",", ":"))
    result = {
        "model": args.model, "model_revision": getattr(model.config, "_commit_hash", None) or model_revision,
        "data_source": data_source, "pairs_sha256": hashlib.sha256(pairs_json.encode("utf-8")).hexdigest(),
        "sample_size": len(questions), "seed": args.seed, "device": str(model.device), "pooling": "CLS",
        "max_length": min(args.max_length, tokenizer.model_max_length, model.config.max_position_embeddings),
        "top_k": min(args.top_k, len(answers)), "top1": top1, "topk": topk,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.output_json:
        args.output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.plot:
        import numpy as np

        all_embeddings = np.vstack((question_embeddings.numpy(), answer_embeddings.numpy()))
        labels = ["question"] * len(questions) + ["answer"] * len(answers)
        plot_tsne(all_embeddings, labels, f"{args.model} t-SNE", args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
