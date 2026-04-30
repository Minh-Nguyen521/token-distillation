from fire import Fire

from token_distillation import DistillationConfig, HFDataSource, SentenceTransformerTokenDistillation
from token_distillation.tokdist_st import DatasetTokenSource, extract_tokens_from_dataset


def main(
    model_path: str = "sentence-transformers/all-MiniLM-L6-v2",
    out_path: str = "./outputs/minilm-vietnamese",
    # new tokens: "list" or "dataset"
    token_source: str = "dataset",
    tokens: str = "",  # comma-separated, used when token_source="list"
    # dataset token extraction (used when token_source="dataset")
    token_dataset_path: str = "bkai-foundation-models/crosslingual",
    token_data_files: str = "original/merged_queries_vi.json,synthetic/cross_queries.parquet",
    top_k: int = 200,
    min_freq: int = 10,
    min_subtokens: int = 2,
    # data for snippet mining
    snippet_dataset_path: str = "bkai-foundation-models/crosslingual",
    snippet_data_files: str = "original/merged_queries_vi.json,synthetic/cross_queries.parquet",
    max_docs: int | None = 500_000,
    # snippets
    snippet_len: int = 50,
    snippets_per_token: int = 100,
    # training
    epochs: int = 1,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    seed: int = 42,
    mixed_precision: bool = True,
    # device
    device: str = "cuda:0",
):
    tokdist = SentenceTransformerTokenDistillation(model_path=model_path, device=device)

    # --- resolve new tokens ---
    if token_source == "list":
        new_tokens = [t.strip() for t in tokens.split(",") if t.strip()]
    elif token_source == "dataset":
        data_files = [f.strip() for f in token_data_files.split(",") if f.strip()]

        # synthetic/cross_queries.parquet has "query" and "pos" columns
        def crosslingual_text(example: dict) -> str:
            query = example.get("query", "")
            pos = example.get("pos", [])
            passage = pos[0] if isinstance(pos, list) and pos else ""
            return f"{query} {passage}".strip()

        new_tokens = extract_tokens_from_dataset(
            DatasetTokenSource(
                dataset_path=token_dataset_path,
                data_files=data_files,
                max_docs=min(max_docs, 200_000) if max_docs else 200_000,
                top_k=top_k,
                min_freq=min_freq,
                min_subtokens=min_subtokens,
                map_to_text_fn=crosslingual_text,
            ),
            tokenizer=tokdist.source_tokenizer,
        )
    else:
        raise ValueError("token_source must be 'list' or 'dataset'")

    print(f"[example_st] {len(new_tokens)} new tokens: {new_tokens[:10]}{'...' if len(new_tokens) > 10 else ''}")

    # --- snippet mining data ---
    s_data_files = [f.strip() for f in snippet_data_files.split(",") if f.strip()]

    def crosslingual_text(example: dict) -> str:
        query = example.get("query", "")
        pos = example.get("pos", [])
        passage = pos[0] if isinstance(pos, list) and pos else ""
        return f"{query} {passage}".strip()

    data = HFDataSource(
        dataset_path=snippet_dataset_path,
        split="train",
        max_docs=max_docs,
        map_to_text_fn=crosslingual_text,
    )

    tokdist.run(
        new_tokens=new_tokens,
        data=data,
        out_path=out_path,
        snippet_len=snippet_len,
        snippets_per_token=snippets_per_token,
        training=DistillationConfig(
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            seed=seed,
            mixed_precision=mixed_precision,
        ),
        save=True,
    )


if __name__ == "__main__":
    Fire(main)
