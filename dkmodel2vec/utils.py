from transformers import AutoTokenizer
from sklearn.decomposition import IncrementalPCA
import numpy as np

from datasets import Dataset
from tqdm import tqdm

import logging

logger = logging.getLogger(__name__)


def check_fits_length(
    batch: dict[str, list],
    tokenizer: AutoTokenizer,
    max_length: int,
    doc_max_length: int,
) -> dict[str, list[bool]]:
    """Check if texts fit within doc_max_length without truncation."""
    texts = batch["document"]

    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )

    fits_length = [
        len(token_ids) <= doc_max_length for token_ids in tokenized["input_ids"]
    ]

    return {"fits_length": fits_length}


def add_instruction_to_text(
    batch: dict[str, list], query_instruction: str
) -> dict[str, list[str]]:
    """Add instruction prefix to texts. Assumes texts already fit length requirements."""
    texts = batch["document"]
    columns = batch["column"]

    processed_texts = []

    for text, column in zip(texts, columns):
        instruction = query_instruction if column == "query" else ""

        if instruction:
            processed_text = f"{instruction.strip()} !@#$%^&*(){text}"
        else:
            processed_text = f"!@#$%^&*(){text}"

        processed_texts.append(processed_text)

    return {"processed_text": processed_texts}


def iterable_dimension_reduction(
    ds: Dataset,
    n_components: int = 50,
    batch_size: int = 50_000,
    column_name: str = "embedding",
) -> Dataset:
    """Apply PCA to dataset embeddings in a memory-efficient manner."""

    ipca = IncrementalPCA(
        n_components=n_components, batch_size=min(batch_size, ds.num_rows)
    )

    logger.info(
        f"Fitting PCA with {n_components} components on {ds.num_rows:,} samples..."
    )

    total_batches = (ds.num_rows + batch_size - 1) // batch_size

    # Fit phase - optimized
    for batch in tqdm(
        ds.iter(batch_size=batch_size), total=total_batches, desc="Fitting PCA"
    ):
        # Direct numpy conversion is faster than list intermediate
        batch_vec = np.stack(batch[column_name], dtype=np.float32)
        ipca.partial_fit(batch_vec)

    logger.info(
        f"PCA fitted. Variance explained: {ipca.explained_variance_ratio_.sum():.2%}"
    )

    # Transform phase - using map is faster
    logger.info("Transforming data...")

    def transform_batch(batch):
        embeddings = np.stack(batch[column_name], dtype=np.float32)
        pca_embeddings = ipca.transform(embeddings)
        return {column_name: pca_embeddings.tolist()}

    ds = ds.map(
        transform_batch, batched=True, batch_size=batch_size, desc="Applying PCA"
    )

    logger.info("Done! PCA transformation complete.")

    return ds
