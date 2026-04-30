import time
from typing import Sequence

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_scheduler

from .train_loop import TextDataset, transform_input_token_format
from .utils import seed_everything

TokenLike = Sequence[int] | torch.Tensor
PhraseKey = tuple[int, ...]
GroupedTokenizedTexts = Sequence[Sequence[TokenLike]]


def collate_fn_st(batch, pad_id):
    max_len = max(len(sample["merged_seq"]) for sample in batch)
    merged_seqs, original_seqs, masks = [], [], []
    merged_attn_masks, original_attn_masks = [], []

    for sample in batch:
        m = list(sample["merged_seq"])
        o = list(sample["original_seq"])
        mask = list(sample["unmerged_to_merged_mask"])
        pad_len = max_len - len(m)
        orig_pad_len = max_len - len(o)

        merged_attn_masks.append([1] * len(m) + [0] * pad_len)
        original_attn_masks.append([1] * len(o) + [0] * orig_pad_len)

        merged_seqs.append(m + [pad_id] * pad_len)
        original_seqs.append(o + [pad_id] * orig_pad_len)
        masks.append(mask + [0] * pad_len)

    return {
        "merged_seq": torch.tensor(merged_seqs, dtype=torch.long),
        "original_seq": torch.tensor(original_seqs, dtype=torch.long),
        "unmerged_to_merged_mask": torch.tensor(masks, dtype=torch.long),
        "merged_attention_mask": torch.tensor(merged_attn_masks, dtype=torch.long),
        "original_attention_mask": torch.tensor(original_attn_masks, dtype=torch.long),
    }


def train_embeddings_st(
    model,
    tokenized_texts: GroupedTokenizedTexts,
    new_phrase_to_new_id: dict[PhraseKey, int],
    assigned_new_phrases: Sequence[TokenLike],
    tokenizer,
    epochs: int = 1,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    seed: int = 42,
    mixed_precision: bool = True,
    preserve_original_embeddings: bool = True,
    original_token_ids: list[int] | None = None,
):
    t0 = time.perf_counter()
    seed_everything(seed)

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        print(f"[tokdist-st] Setting pad token to eos: {tokenizer.eos_token}")

    dataset = TextDataset(
        transform_input_token_format(
            tokenized_texts,
            new_phrase_to_new_id,
            tokenizer.pad_token_id,
            assigned_new_phrases=assigned_new_phrases,
        )
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0,
        drop_last=True,
        collate_fn=lambda b: collate_fn_st(b, pad_id=tokenizer.pad_token_id),
    )

    inner_model = model[0].auto_model

    for p in model.parameters():
        p.requires_grad = False
    inner_model.get_input_embeddings().weight.requires_grad = True

    original_input_embs = inner_model.get_input_embeddings().weight.clone().detach()

    if original_token_ids is None:
        original_token_ids = list(range(len(tokenizer) - len(new_phrase_to_new_id)))

    optimizer = AdamW([inner_model.get_input_embeddings().weight], lr=learning_rate, weight_decay=0.0)
    optimizer.zero_grad()
    scheduler = get_scheduler("constant", optimizer=optimizer)
    model.train()

    device = next(model.parameters()).device
    print(model)
    print(f"[tokdist-st] Training startup time: {time.perf_counter() - t0:.2f}s")

    for epoch in range(epochs):
        epoch_bar = tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Epoch: {epoch}")
        running_window_losses = []

        for step_idx, batch in epoch_bar:
            merged_seq = batch["merged_seq"].to(device, non_blocking=True)
            original_seq = batch["original_seq"].to(device, non_blocking=True)
            merged_attn = batch["merged_attention_mask"].to(device, non_blocking=True)
            original_attn = batch["original_attention_mask"].to(device, non_blocking=True)

            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=mixed_precision):
                student_out = model({"input_ids": merged_seq, "attention_mask": merged_attn})
                with torch.no_grad():
                    teacher_out = model({"input_ids": original_seq, "attention_mask": original_attn})

            loss = F.mse_loss(
                student_out["sentence_embedding"].float(),
                teacher_out["sentence_embedding"].float(),
            )

            loss.backward(inputs=[inner_model.get_input_embeddings().weight])

            if preserve_original_embeddings:
                inner_model.get_input_embeddings().weight.grad[original_token_ids] = 0

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            running_window_losses.append(loss.item())
            avg_loss = sum(running_window_losses) / len(running_window_losses)
            epoch_bar.set_description(f"Epoch: {epoch}, Loss: {loss.item():.6f}, Running: {avg_loss:.6f}")

            if len(running_window_losses) == int(len(dataloader) * 0.1):
                running_window_losses = []
                print(f"Epoch: {epoch}, Step: {step_idx}, Loss: {loss.item():.6f}, Running: {avg_loss:.6f}")

    if preserve_original_embeddings:
        if not torch.equal(
            inner_model.get_input_embeddings().weight.data[original_token_ids],
            original_input_embs[original_token_ids],
        ):
            raise RuntimeError("Original input embeddings have changed.")

    print(f"[tokdist-st] Total training time: {time.perf_counter() - t0:.2f}s")
    return model
