from __future__ import annotations

import collections
import copy
import json
import os
import re
import tempfile
from dataclasses import dataclass, field
from typing import Callable, List, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm

from .tokdist import (
    GPT_BPE_WHITESPACE,
    SPIECE_WHITESPACE,
    HFDataSource,
    DistillationConfig,
    detect_whitespace_token,
    build_snippets_for_tokens_from_hf,
)
from .train_loop_st import train_embeddings_st
from .utils import get_new_phrase_tokenized_ids


@dataclass
class DatasetTokenSource:
    """Automatically extract new token candidates from a dataset.

    Scans the dataset, finds words that are split into 2+ subwords by the
    current tokenizer, and returns the most frequent ones as new token candidates.

    Args:
        dataset_path: HuggingFace dataset path.
        name: Dataset config name (e.g. "deu_Latn").
        split: Dataset split (e.g. "train").
        max_docs: Number of documents to scan.
        top_k: Number of most frequent multi-subword words to return as new tokens.
        min_freq: Minimum frequency for a word to be considered.
        min_subtokens: Only consider words split into at least this many subwords.
        data_files: List of HF-relative file paths to load (e.g. ["original/merged_queries_vi.json"]).
            When set, overrides split-based loading.
        map_to_text_fn: Optional function to extract a text string from each dataset example dict.
            Defaults to using the "text" or "query" column.
    """

    dataset_path: str
    name: str | None = None
    split: str = "train"
    max_docs: int = 100_000
    top_k: int = 100
    min_freq: int = 10
    min_subtokens: int = 2
    data_files: list[str] | None = None
    map_to_text_fn: Callable | None = None


def _load_texts_from_file(path: str, map_to_text_fn: Callable | None) -> list[str]:
    """Load text strings from a single local JSON or Parquet file."""
    if path.endswith(".json"):
        with open(path) as f:
            data = json.load(f)
        if isinstance(data, dict):
            # {id: text} format (e.g. merged_queries_vi.json)
            return [str(v) for v in data.values()]
        elif isinstance(data, list):
            if map_to_text_fn:
                return [map_to_text_fn(item) for item in data]
            return [item.get("text") or item.get("query", "") for item in data]
    elif path.endswith(".parquet"):
        import pandas as pd
        df = pd.read_parquet(path)
        if map_to_text_fn:
            return [map_to_text_fn(row.to_dict()) for _, row in df.iterrows()]
        for col in ["text", "query"]:
            if col in df.columns:
                return df[col].tolist()
    raise ValueError(f"Unsupported file format or missing text column: {path}")


def _load_texts_from_hf(source: DatasetTokenSource) -> list[str]:
    """Load texts from HuggingFace dataset, handling data_files if provided."""
    texts = []
    files = source.data_files if source.data_files else [None]

    for data_file in files:
        kwargs = dict(streaming=True)
        if data_file:
            kwargs["data_files"] = data_file
            kwargs["split"] = "train"
        else:
            kwargs["split"] = source.split
            if source.name:
                kwargs["name"] = source.name

        ds = load_dataset(source.dataset_path, **kwargs)
        for i, example in enumerate(ds):
            if i >= source.max_docs:
                break
            if source.map_to_text_fn:
                text = source.map_to_text_fn(example)
            else:
                text = example.get("text") or example.get("query", "")
            if text:
                texts.append(text)

    return texts


def extract_tokens_from_dataset(source: DatasetTokenSource, tokenizer) -> List[str]:
    """Scan a dataset and return frequent words that are split into multiple subwords."""
    print(f"[tokdist-st] Scanning up to {source.max_docs} docs from {source.dataset_path} for token candidates...")

    texts = _load_texts_from_hf(source)

    word_freq: collections.Counter = collections.Counter()
    for text in tqdm(texts, desc="Counting words"):
        words = re.findall(r"[^\s\.,!?;:()\[\]{}'\"]+", text)
        word_freq.update(w for w in words if len(w) > 2)

    # keep only words that tokenize to >= min_subtokens subwords
    candidates = []
    for word, freq in word_freq.most_common():
        if freq < source.min_freq:
            break
        token = f" {word}"  # leading space = mid-sentence context
        subwords = tokenizer.tokenize(token)
        if len(subwords) >= source.min_subtokens:
            candidates.append(token)
        if len(candidates) >= source.top_k:
            break

    print(f"[tokdist-st] Found {len(candidates)} token candidates from dataset.")
    return candidates

    # keep only words that tokenize to >= min_subtokens subwords
    candidates = []
    for word, freq in word_freq.most_common():
        if freq < source.min_freq:
            break
        token = f" {word}"  # leading space = mid-sentence context
        subwords = tokenizer.tokenize(token)
        if len(subwords) >= source.min_subtokens:
            candidates.append(token)
        if len(candidates) >= source.top_k:
            break

    print(f"[tokdist-st] Found {len(candidates)} token candidates from dataset.")
    return candidates


class SentenceTransformerTokenDistillation:
    """Token Distillation for SentenceTransformer models.

    Learns input embeddings for new tokens by distilling the pooled sentence
    embedding produced by the original multi-token representation into a single
    new token embedding. Only input embeddings are updated; there is no LM head.
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda:0",
        trust_remote_code: bool = False,
        local_files_only: bool = False,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise RuntimeError("SentenceTransformerTokenDistillation requires CUDA.")
        if not device.startswith("cuda"):
            raise ValueError(f"A CUDA device string is required, got {device!r}.")

        self.model_path = model_path
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.local_files_only = local_files_only

        self.model = SentenceTransformer(
            model_path,
            device=device,
            trust_remote_code=trust_remote_code,
            local_files_only=local_files_only,
        )
        self.inner_model = self.model[0].auto_model
        self.source_tokenizer = self.model.tokenizer
        self._src_ws = detect_whitespace_token(self.source_tokenizer)
        self._tokenizer_repr = model_path.split("/")[-1]

    def run(
        self,
        new_tokens: List[str],
        data: HFDataSource,
        out_path: str | None = None,
        snippet_len: int = 50,
        snippets_per_token: int = 100,
        training: DistillationConfig = DistillationConfig(),
        save: bool = False,
    ) -> Tuple:
        """Add new tokens, distill embeddings, and optionally save the model.

        Args:
            new_tokens: List of new token strings to add to the vocabulary.
            data: HFDataSource config for mining context snippets.
            out_path: Directory to save the updated model. Required if save=True.
            snippet_len: Number of tokens per context snippet.
            snippets_per_token: Number of snippets to collect per new token.
            training: DistillationConfig (epochs, batch_size, learning_rate, seed,
                mixed_precision are used; loss_methods and target_layer are ignored).
            save: Whether to save the model and tokenizer to out_path.

        Returns:
            (model, tokenizer) tuple.
        """
        if out_path is None and save:
            raise ValueError("out_path must be specified if save=True.")
        if out_path is None:
            out_path = tempfile.mkdtemp(prefix="tokdist_st_")
            print(f"[tokdist-st] Using temp path: {out_path}")

        original_vocab_size = len(self.source_tokenizer)
        todo_tokens = self._build_target_tokenizer(new_tokens)
        if not todo_tokens:
            print("[tokdist-st] No new tokens to add.")
            return self.model, self.source_tokenizer

        # 1) collect snippets from HF dataset
        new_phrases_ids, new_phrases_snippets_ids, skipped = build_snippets_for_tokens_from_hf(
            self.model_path,
            self.source_tokenizer,
            todo_tokens,
            snippet_len,
            snippets_per_token,
            data,
            tokenizer_repr=self._tokenizer_repr,
            trust_remote_code=self.trust_remote_code,
        )

        if skipped:
            print(f"[tokdist-st] Skipped {len(skipped)} tokens due to insufficient snippets:")
            for token, _, available in skipped:
                print(f"  - {token!r}: {available}/{snippets_per_token} snippets")
            skipped_ids = {token_id for _, token_id, _ in skipped}
            todo_tokens = [t for t in todo_tokens if t[1] not in skipped_ids]
            if not todo_tokens:
                raise RuntimeError("No tokens left to distill after filtering skipped tokens.")

        # 2) pre-init new embeddings via subtoken mean
        input_preinit = self._compute_subtoken_means(todo_tokens)

        # 3) extend tokenizer and resize embedding table
        self.source_tokenizer.add_tokens(list(input_preinit.keys()))
        self.inner_model.resize_token_embeddings(len(self.source_tokenizer))

        in_w = self.inner_model.get_input_embeddings().weight.data
        for token, emb in input_preinit.items():
            token_id = self.source_tokenizer.convert_tokens_to_ids(token)
            in_w[token_id] = emb
        self.inner_model.get_input_embeddings().weight.data = in_w

        # 4) train input embeddings via MSE on sentence embeddings
        assigned_new_phrases = [tuple(phrase_ids.tolist()) for phrase_ids in new_phrases_ids]
        phrase_to_new_id = {phrase: original_vocab_size + i for i, phrase in enumerate(assigned_new_phrases)}

        self.model = train_embeddings_st(
            self.model,
            new_phrases_snippets_ids,
            phrase_to_new_id,
            assigned_new_phrases,
            tokenizer=self.source_tokenizer,
            epochs=training.epochs,
            batch_size=training.batch_size,
            learning_rate=training.learning_rate,
            seed=training.seed,
            mixed_precision=training.mixed_precision,
        )

        # 5) save
        if save:
            os.makedirs(out_path, exist_ok=True)
            self.model.save(out_path)
            self.source_tokenizer.save_pretrained(out_path)
            print(f"[tokdist-st] Saved model and tokenizer to {out_path}")

        return self.model, self.source_tokenizer

    def _build_target_tokenizer(self, new_tokens: List[str]) -> List[Tuple[str, int]]:
        """Filter tokens that are genuinely new and compute their future IDs."""
        src_vocab = self.source_tokenizer.get_vocab()
        filtered, seen = [], set()
        for token in new_tokens:
            if token in seen:
                continue
            seen.add(token)
            token_norm = token.replace(GPT_BPE_WHITESPACE, self._src_ws).replace(SPIECE_WHITESPACE, self._src_ws)
            if src_vocab.get(token_norm.replace(" ", self._src_ws)) is not None:
                continue
            if token in self.source_tokenizer.all_special_tokens:
                continue
            if token.startswith("<0x") and token.endswith(">"):
                continue
            filtered.append(token)

        # use a copy to resolve future IDs without modifying the real tokenizer yet
        temp_tokenizer = copy.deepcopy(self.source_tokenizer)
        if filtered:
            temp_tokenizer.add_tokens(filtered)
        todo_tokens = [(token, temp_tokenizer.convert_tokens_to_ids(token)) for token in filtered]
        print(f"[tokdist-st] {len(todo_tokens)} tokens need initialization")
        return todo_tokens

    @torch.no_grad()
    def _compute_subtoken_means(self, todo_tokens: List[Tuple[str, int]]) -> dict[str, torch.Tensor]:
        """Pre-initialize new token embeddings as the mean of their subtoken embeddings."""
        embs = self.inner_model.get_input_embeddings().weight
        out = {}
        for token, _ in todo_tokens:
            token_ids = get_new_phrase_tokenized_ids(token, self.source_tokenizer, self.model_path).to(embs.device)
            out[token] = torch.mean(embs[token_ids], dim=0)
        return out
