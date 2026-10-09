import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from transformers import BertConfig, BertModel, BertTokenizer

import ana_kod_tugcantopaloglu as retrieval


class DataTests(unittest.TestCase):
    def test_remote_assets_require_immutable_revisions_and_local_paths_do_not(self):
        self.assertEqual(retrieval._asset_revision(retrieval.MODEL_NAME, None, retrieval.MODEL_NAME, retrieval.MODEL_REVISION), retrieval.MODEL_REVISION)
        for revision in (None, "main", "v1", "abcdef0", "z" * 40):
            with self.subTest(revision=revision), self.assertRaises(ValueError):
                retrieval._asset_revision("other/model", revision, retrieval.MODEL_NAME, retrieval.MODEL_REVISION)
        with tempfile.TemporaryDirectory() as directory:
            self.assertIsNone(retrieval._asset_revision(directory, None, retrieval.MODEL_NAME, retrieval.MODEL_REVISION))

    def test_spaced_turkish_columns_and_repeatable_sample_keep_pairs_aligned(self):
        records = [
            {"talimat": "Başkent?", " giriş": "Türkiye", " çıktı": "Ankara"},
            {"talimat": "İl?", "giriş": None, "çıktı": "İzmir"},
            {"talimat": "Dil?", "giriş": "", "çıktı": "Türkçe"},
        ]
        selected = retrieval.select_question_answer_pairs(records, sample_size=3, seed=7)
        self.assertEqual(selected, retrieval.select_question_answer_pairs(records, 3, 7))
        self.assertEqual(set(zip(*selected)), {("Başkent? Türkiye", "Ankara"), ("İl?", "İzmir"), ("Dil?", "Türkçe")})

    def test_arrow_dataset_with_null_inputs_retains_rows(self):
        from datasets import Dataset

        dataset = Dataset.from_dict({"talimat": ["Question one", "Question two"], " giriş": [None, None], " çıktı": ["Answer one", "Answer two"]})
        questions, answers = retrieval.select_question_answer_pairs(dataset, sample_size=2)
        self.assertEqual(set(zip(questions, answers)), {("Question one", "Answer one"), ("Question two", "Answer two")})

    def test_sample_bounds_and_invalid_rows_fail_before_inference(self):
        records = [{"talimat": "Question", "çıktı": "Answer"}]
        for sample_size in (0, 2):
            with self.subTest(sample_size=sample_size), self.assertRaises(ValueError):
                retrieval.select_question_answer_pairs(records, sample_size)
        for record in (
            {"talimat": "", "çıktı": "Answer"},
            {"talimat": "Question", "çıktı": None},
            {"talimat": "Question", "çıktı": "Answer", " çıktı": "Other"},
        ):
            with self.subTest(record=record), self.assertRaises(ValueError):
                retrieval.select_question_answer_pairs([record], 1)

    def test_csv_preserves_turkish_commas_newlines_and_literal_na(self):
        questions = ["İstanbul, Türkiye\nSoru", "NA"]
        answers = ["Çıktı\nİzmir", "null"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pairs.csv"
            retrieval.save_question_answer_csv(path, questions, answers)
            self.assertEqual(retrieval.load_question_answer_csv(path), (questions, answers))
            path.write_text("Question,Answer\nQuestion,\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                retrieval.load_question_answer_csv(path)
            path.write_text("prompt,response\nQuestion,Answer\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                retrieval.load_question_answer_csv(path)
            for malformed in ("Question,Answer,Answer\nQuestion,Answer,Other\n", "Question,Answer\nQuestion,Answer,Extra\n"):
                path.write_text(malformed, encoding="utf-8")
                with self.assertRaises(ValueError):
                    retrieval.load_question_answer_csv(path)

    def test_import_and_help_need_no_ml_dependencies(self):
        import importlib.util

        spec = importlib.util.spec_from_file_location("retrieval_import_test", retrieval.__file__)
        module = importlib.util.module_from_spec(spec)
        with patch.dict("sys.modules", {"torch": None, "transformers": None, "datasets": None}):
            spec.loader.exec_module(module)
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exited:
                module.main(["--help"])
        self.assertEqual(exited.exception.code, 0)


class RankingTests(unittest.TestCase):
    def test_top_k_uses_cosine_and_paired_indices(self):
        questions = [[2, 0], [0, 4], [-5, 0]]
        answers = [[3, 0], [-2, 0], [0, 9]]
        self.assertEqual(retrieval.evaluate_embeddings(questions, answers, k=1, batch_size=2), (1 / 3, 1 / 3))
        self.assertEqual(retrieval.evaluate_embeddings(questions, answers, k=2, batch_size=1), (1 / 3, 2 / 3))
        self.assertEqual(retrieval.evaluate_embeddings(questions, answers, k=5), (1 / 3, 1.0))
        self.assertAlmostEqual(retrieval.calculate_similarity([2, 0], [100, 0]), 1.0)
        self.assertAlmostEqual(retrieval.calculate_similarity([2, 0], [0, 100]), 0.0)

    def test_ties_use_candidate_row_order_and_duplicates_remain_separate(self):
        duplicates = [[1, 0], [1, 0], [1, 0]]
        self.assertEqual(retrieval.evaluate_embeddings(duplicates, duplicates, k=2), (1 / 3, 2 / 3))

    def test_invalid_embeddings_do_not_produce_misleading_metrics(self):
        for values in ([], [[0, 0]], [[np.nan, 1]], [[np.inf, 1]], [1, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                retrieval.evaluate_embeddings(values, values)
        with self.assertRaises(ValueError):
            retrieval.evaluate_embeddings([[1, 0]], [[1, 0], [0, 1]])
        with self.assertRaises(ValueError):
            retrieval.evaluate_embeddings([[1, 0]], [[1, 0]], k=0)


class LocalBertTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.directory = tempfile.TemporaryDirectory()
        cls.model_path = Path(cls.directory.name) / "tiny-bert"
        cls.model_path.mkdir()
        vocabulary = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "Ankara", "İzmir", "Türkiye", "Başkent", "Dil", "Türkçe", "?"]
        vocab_path = cls.model_path / "vocab.txt"
        vocab_path.write_text("\n".join(vocabulary) + "\n", encoding="utf-8")
        tokenizer = BertTokenizer(vocab={token: index for index, token in enumerate(vocabulary)}, do_lower_case=False, model_max_length=12)
        tokenizer.save_pretrained(cls.model_path)
        torch.manual_seed(7)
        config = BertConfig(
            vocab_size=len(vocabulary), hidden_size=8, num_hidden_layers=1,
            num_attention_heads=2, intermediate_size=16, max_position_embeddings=12,
        )
        BertModel(config).save_pretrained(cls.model_path)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def setUp(self):
        self.tokenizer, self.model = retrieval.get_model_and_tokenizer(str(self.model_path), "cpu", local_files_only=True)

    def test_real_local_tokenizer_cpu_cls_batches_and_dropout(self):
        self.assertEqual(self.tokenizer.tokenize("Ankara"), ["Ankara"])
        self.assertFalse(self.model.training)
        self.model.train()
        texts = ["Ankara", "İzmir Türkiye", "Başkent " * 20]
        forward_inputs = []

        def record_inputs(model, args, kwargs):
            forward_inputs.append((kwargs["input_ids"].shape, model.training, torch.is_grad_enabled()))

        hook = self.model.register_forward_pre_hook(record_inputs, with_kwargs=True)
        try:
            batched = retrieval.get_embeddings_in_batches(texts, self.tokenizer, self.model, batch_size=2)
        finally:
            hook.remove()
        single = retrieval.get_embeddings(texts, self.tokenizer, self.model)
        self.assertEqual(batched.shape, (3, 8))
        self.assertEqual(str(batched.device), "cpu")
        self.assertFalse(batched.requires_grad)
        self.assertEqual([shape[0] for shape, _, _ in forward_inputs], [2, 1])
        self.assertEqual(forward_inputs[-1][0][1], 12)
        self.assertTrue(all(not training and not grad for _, training, grad in forward_inputs))
        torch.testing.assert_close(batched, single, atol=1e-6, rtol=1e-5)
        reference_inputs = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=12)
        self.assertTrue((reference_inputs["input_ids"][:, 0] == self.tokenizer.cls_token_id).all())
        with torch.inference_mode():
            reference = self.model(**reference_inputs).last_hidden_state[:, 0, :]
        torch.testing.assert_close(batched, reference, atol=1e-6, rtol=1e-5)
        for batch_size in (0, -1):
            with self.assertRaises(ValueError):
                retrieval.get_embeddings_in_batches(texts, self.tokenizer, self.model, batch_size=batch_size)

    def test_auto_device_falls_back_to_cpu(self):
        with patch("torch.cuda.is_available", return_value=False):
            _, model = retrieval.get_model_and_tokenizer(str(self.model_path), local_files_only=True)
        self.assertEqual(str(model.device), "cpu")

    def test_unavailable_cuda_fails_before_loading_weights(self):
        with patch("torch.cuda.is_available", return_value=False), patch("transformers.AutoModel.from_pretrained") as loader:
            with self.assertRaisesRegex(ValueError, "CUDA was requested"):
                retrieval.get_model_and_tokenizer(str(self.model_path), device="cuda", local_files_only=True)
        loader.assert_not_called()

    def test_answer_candidates_are_encoded_once_in_bounded_batches(self):
        counts = []

        def count_batch(model, args, kwargs):
            counts.append(kwargs["input_ids"].shape[0])

        hook = self.model.register_forward_pre_hook(count_batch, with_kwargs=True)
        try:
            top1, topk = retrieval.calculate_top_k_success(
                ["Ankara", "İzmir", "Türkçe"], ["Türkiye", "Başkent", "Dil"],
                self.tokenizer, self.model, batch_size=2,
            )
        finally:
            hook.remove()
        self.assertEqual(counts, [2, 1, 2, 1])
        self.assertGreaterEqual(top1, 0)
        self.assertLessEqual(top1, topk)
        self.assertEqual(topk, 1)

    def test_cli_runs_with_local_csv_model_and_records_real_fixture_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            csv_path = Path(directory) / "pairs.csv"
            result_path = Path(directory) / "results.json"
            retrieval.save_question_answer_csv(csv_path, ["Ankara", "İzmir"], ["Türkiye", "Başkent"])
            with contextlib.redirect_stdout(io.StringIO()):
                code = retrieval.main([
                    "--csv", str(csv_path), "--model", str(self.model_path), "--device", "cpu",
                    "--local-files-only", "--batch-size", "1", "--output-json", str(result_path),
                ])
            result = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertEqual(code, 0)
            self.assertEqual(result["sample_size"], 2)
            self.assertEqual(result["device"], "cpu")
            self.assertEqual(result["max_length"], 12)
            self.assertEqual(result["top_k"], 2)
            self.assertEqual(result["topk"], 1)
            self.assertEqual(len(result["pairs_sha256"]), 64)


class PlotTests(unittest.TestCase):
    def test_small_fixture_uses_valid_perplexity_and_saves_plot(self):
        import matplotlib

        matplotlib.use("Agg")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plot.png"
            retrieval.plot_tsne(
                np.array([[1, 0, 1], [0, 1, 1], [1, 1, 0], [-1, 1, 1]], dtype=np.float32),
                ["question", "question", "answer", "answer"], "Synthetic fixture", path,
            )
            self.assertTrue(path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
