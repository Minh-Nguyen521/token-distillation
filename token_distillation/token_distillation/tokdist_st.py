from __future__ import annotations

import copy
import os
import tempfile
from typing import List, Tuple

import torch

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


class SentenceTransformerTokenDistillation:
    """Token Distillation for SentenceTransformer models.

    Learns input embeddings for new tokens by distilling the pooled sentence
    embedding produced by the original multi-token representation into a single
    new token embedding. Only input embeddings are updated; there is no LM head.
    """

    def __init__(self, model_path: str, device: str = "cuda:0") -> None:
        from sentence_transformers import SentenceTransformer

        if not torch.cuda.is_available():
            raise RuntimeError("SentenceTransformerTokenDistillation requires CUDA.")
        if not device.startswith("cuda"):
            raise ValueError(f"A CUDA device string is required, got {device!r}.")

        self.model_path = model_path
        self.device = device

        self.model = SentenceTransformer(model_path, device=device)
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
