import numpy as np
import pandas as pd
import mlflow

from model2vec import StaticModel
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.metrics import confusion_matrix, classification_report

# import matplotlib.pyplot as plt
# import seaborn as sns
from tqdm import tqdm
from typing import Dict, Tuple
from datasets import DatasetDict, Dataset
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from dkmodel2vec.config import BEST_SENTENCE_TRANSFORMER, DANISH_INSTRUCTION
from dkmodel2vec.constants import (
    PREDICTION_COLUMN,
    BM25_PREDICTION_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
    DATASET_QUERY_COLUMN,
    LLM2VEC_PREDICTION_COLUMN
)
from llm2vec.llm2vec import LLM2Vec

from logging import getLogger

logger = getLogger(__name__)


def load_sentence_transformer(model_name: str = BEST_SENTENCE_TRANSFORMER):
    model = SentenceTransformer(model_name)
    model.half()  # reduce precision to speed up inference
    return model


def predict_bm25(example: dict, tokenizer: Tokenizer) -> np.array:
    """Convert a query its corresponding tokens and get the bm25 prediction from the score across the corpus consisting of a single positive and negative candidates."""

    # Use backend_tokenizer to get Encoding objects with .tokens attribute
    q_encoding = tokenizer.backend_tokenizer.encode(example["query"])
    q_tokens = q_encoding.tokens

    corpus = []
    for text in [example["negative"], example["positive"]]:
        if isinstance(text, str):
            encoding = tokenizer.backend_tokenizer.encode(text)
            corpus.append(encoding.tokens)
        else:
            example[BM25_PREDICTION_COLUMN] = -1
            return example

    # somehow BM25 does not work with only 2 examples? so adding a third
    corpus.append(
        [
            "ULTRARARETOKEN",
            "ULTRARARETOKEN2",
            "ULTRARARETOKEN3",
            "ULTRARARETOKEN4",
            "ULTRARARETOKEN5",
        ]
    )

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tokens)[:2]  # get rid of the dummy document
    example[BM25_PREDICTION_COLUMN] = np.argmax(scores)

    return example


def predict_sentence_transformer_cos_sim(
    batch: dict,
    sentence_transformer: SentenceTransformer,
    out_column: str = "sentence_transformer_pred",
) -> dict:
    """Predict the most similar document index (0 or 1) using cosine similarity."""
    # Convert to numpy arrays for boolean indexing
    queries = batch[DATASET_QUERY_COLUMN]
    positives = batch[DATASET_POSITIVE_COLUMN]
    negatives = batch[DATASET_NEGATIVE_COLUMN]

    query_embeds = sentence_transformer.encode(queries)
    pos_embeds = sentence_transformer.encode(positives)
    neg_embeds = sentence_transformer.encode(negatives)

    # Calculate cosine similarities
    pos_similarities = np.sum(query_embeds * pos_embeds, axis=1) / (
        np.linalg.norm(query_embeds, axis=1) * np.linalg.norm(pos_embeds, axis=1)
    )
    neg_similarities = np.sum(query_embeds * neg_embeds, axis=1) / (
        np.linalg.norm(query_embeds, axis=1) * np.linalg.norm(neg_embeds, axis=1)
    )

    # Higher cosine similarity means more similar (opposite of distance)
    predictions = (pos_similarities > neg_similarities).astype(int)

    batch[out_column] = predictions.tolist()
    return batch

def predict_llm2vec(
    batch: dict,
    model: LLM2Vec,
    out_column: str = LLM2VEC_PREDICTION_COLUMN,
) -> dict:
    """Predict the most similar document index (0 or 1) using model with .encode method."""
    queries = batch[DATASET_QUERY_COLUMN]
    positives = batch[DATASET_POSITIVE_COLUMN]
    negatives = batch[DATASET_NEGATIVE_COLUMN]

    instructions = len(queries)*[DANISH_INSTRUCTION]

    query_embeds = model.encode([[inst_n, q_n] for inst_n, q_n in zip(instructions, queries)])
    pos_embeds = model.encode(positives)
    neg_embeds = model.encode(negatives)

    pos_distances = np.linalg.norm(query_embeds - pos_embeds, axis=1)
    neg_distances = np.linalg.norm(query_embeds - neg_embeds, axis=1)
    predictions = (pos_distances < neg_distances).astype(int)

    batch[out_column] = predictions.tolist()
    return batch

def predict_sentence_transformer(
    batch: dict,
    sentence_transformer: SentenceTransformer,
    out_column: str = "sentence_transformer_pred",
) -> dict:
    """Predict the most similar document index (0 or 1) using model with .encode method."""
    # Convert to numpy arrays for boolean indexing
    queries = batch[DATASET_QUERY_COLUMN]
    positives = batch[DATASET_POSITIVE_COLUMN]
    negatives = batch[DATASET_NEGATIVE_COLUMN]

    query_embeds = sentence_transformer.encode(queries)
    pos_embeds = sentence_transformer.encode(positives)
    neg_embeds = sentence_transformer.encode(negatives)

    pos_distances = np.linalg.norm(query_embeds - pos_embeds, axis=1)
    neg_distances = np.linalg.norm(query_embeds - neg_embeds, axis=1)
    predictions = (pos_distances < neg_distances).astype(int)

    batch[out_column] = predictions.tolist()
    return batch


#    def load_dataset(self, dataset_name: str, split: str = "test",
#                     query_col: str = "query", pos_col: str = "positive",
#                     neg_col: str = "negative") -> pd.DataFrame:
#         """Load and prepare the dataset."""
#         dataset = load_dataset(dataset_name, split=split)
#         df = pd.DataFrame(dataset)

#         # Ensure required columns exist
#         required_cols = [query_col, pos_col, neg_col]
#         missing_cols = [col for col in required_cols if col not in df.columns]
#         if missing_cols:
#             raise ValueError(f"Missing columns: {missing_cols}")

#         return df[[query_col, pos_col, neg_col]].rename(columns={
#             query_col: "query", pos_col: "positive", neg_col: "negative"
#         })


def compute_distances(
    encoder: StaticModel,
    dataset: DatasetDict,
    instruction_encoder: StaticModel | None = None,
    query_column_name="query",
    batch_size: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Euclidean distances between queries and positive/negative examples."""
    if instruction_encoder is None:
        instruction_encoder = encoder

    queries = dataset[query_column_name]
    positives = dataset[DATASET_POSITIVE_COLUMN]
    negatives = dataset[DATASET_NEGATIVE_COLUMN]
    n_samples = len(queries)

    pos_distances = np.zeros(n_samples)
    neg_distances = np.zeros(n_samples)

    for i in tqdm(range(0, n_samples, batch_size), desc="Computing embeddings"):
        batch_end = min(i + batch_size, n_samples)

        batch_queries = queries[i:batch_end]
        batch_positives = positives[i:batch_end]
        batch_negatives = negatives[i:batch_end]

        query_embeds = instruction_encoder.encode(batch_queries)
        pos_embeds = encoder.encode(batch_positives)
        neg_embeds = encoder.encode(batch_negatives)

        # Compute Euclidean distances
        # For distance metrics, smaller values indicate higher similarity
        batch_pos_distances = np.linalg.norm(query_embeds - pos_embeds, axis=1)
        batch_neg_distances = np.linalg.norm(query_embeds - neg_embeds, axis=1)

        pos_distances[i:batch_end] = batch_pos_distances
        neg_distances[i:batch_end] = batch_neg_distances

    return pos_distances, neg_distances


def evaluate_classification(predictions: np.array, ground_truth) -> Dict:
    """Evaluate the classification performance."""

    # Ground truth: always positive class (since we expect pos > neg)
    ground_truth = np.ones_like(predictions)

    # Calculate metrics
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        ground_truth, predictions, average="binary"
    )

    # MCC is not calculated because there is only a single label. Not sure how to think about this.
    #    mcc = matthews_corrcoef(ground_truth, predictions)

    conf_matrix = confusion_matrix(ground_truth, predictions)

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "confusion_matrix": conf_matrix,
        "predictions": predictions,
        "ground_truth": ground_truth,
    }


def log_performance(results: dict, log_prefix: str = ""):
    """Log metric, dicts and table."""
    metrics_to_log = ["accuracy", "precision", "recall", "f1_score"]

    for metric in metrics_to_log:
        if not np.isnan(results[metric]):
            mlflow.log_metric(f"{log_prefix}_{metric}", results[metric])

    class_report = classification_report(
        results["ground_truth"], results["predictions"], output_dict=True
    )
    mlflow.log_dict(class_report, f"{log_prefix}_classification_report.json")

    conf_matrix_df = pd.DataFrame(
        results["confusion_matrix"],
        index=["Actual_0", "Actual_1"],
        columns=["Pred_0", "Pred_1"],
    )
    mlflow.log_table(conf_matrix_df, f"{log_prefix}_confusion_matrix.json")

    return


def evaluate_model(
    dataset: Dataset, model: StaticModel, instruction_model: StaticModel, log_perf=True
) -> Dataset:
    """Run the complete evaluation on the subset of the test set that contains both positive and negative examples."""

    ##### LLM2VEC2MODEL2VEC without instruction
    # Compute similarities
    logger.info("Computing similarities with raw documents")
    pos_dists, neg_dists = compute_distances(encoder=model, dataset=dataset)

    logger.info("Evaluating classification performance...")
    # Classification: positive class if pos_distance < neg_distance
    predictions = (pos_dists < neg_dists).astype(int)
    dataset = dataset.add_column(PREDICTION_COLUMN, predictions)
    results = evaluate_classification(
        predictions, ground_truth=np.ones_like(predictions)
    )
    if log_perf:
        log_performance(results, log_prefix="raw")

    #### LLM2VEC2MODEL2VEC WITH instructions

    logger.info("Computing similarities with documents using instructions...")
    pos_dists, neg_dists = compute_distances(
        encoder=model, instruction_encoder=instruction_model, dataset=dataset
    )

    logger.info("Evaluating classification performance...")
    # Classification: positive class if pos_distance < neg_distance
    predictions_with_instruct = (pos_dists < neg_dists).astype(int)
    dataset = dataset.add_column(
        "prediction_with_instruction", predictions_with_instruct
    )

    results = evaluate_classification(
        predictions_with_instruct, ground_truth=np.ones_like(predictions_with_instruct)
    )
    log_performance(results, log_prefix="instruct")
    return dataset


def evaluate_bm25(dataset: Dataset):
    """BM25 performance with frequencies from positive and negative pair. """
    predictions = np.array(dataset[BM25_PREDICTION_COLUMN])
    bm25_results = evaluate_classification(
        predictions, ground_truth=np.ones_like(predictions)
    )
    log_performance(bm25_results, log_prefix="bm25")
    return

def evaluate_full_bm25(dataset: Dataset):
    """BM25 performance with frequencies from all positive and negative pairs."""
    # TODO: Work in progress
    predictions = np.array(dataset[BM25_PREDICTION_COLUMN])
    bm25_results = evaluate_classification(
        predictions, ground_truth=np.ones_like(predictions)
    )
    log_performance(bm25_results, log_prefix="bm25")
    return

def evaluate_llm2vec_model(dataset: Dataset, model: LLM2Vec)->Dataset:
    """Add predictions for raw llm2vec model as seperate column in dataset and log performance."""
    logger.info("Computing scores with raw llm2vec model... ")
    dataset = dataset.map(
        lambda batch: predict_llm2vec(
            batch, model
        ),
        batched=True,
        batch_size=100,
    )
    predictions = dataset[LLM2VEC_PREDICTION_COLUMN]

    results = evaluate_classification(
        predictions,
        ground_truth=np.ones_like(predictions),
    )
    log_performance(
        results,
        log_prefix="LLM2Vec",
    )
    return dataset

def evaluate_sentence_transformer(
    dataset: Dataset, model_name: str = BEST_SENTENCE_TRANSFORMER
) -> Dataset:
    """Add predictions as seperate column in dataset and log performance."""
    logger.info("Computing scores with sentence transformer model... ")
    sentence_transformer = load_sentence_transformer(model_name)
    out_column_name = model_name
    dataset = dataset.map(
        lambda batch: predict_sentence_transformer(
            batch, sentence_transformer, out_column=out_column_name
        ),
        batched=True,
        batch_size=16384,
    )

    sentence_transformer_predictions = dataset[out_column_name]

    sentence_transformer_results = evaluate_classification(
        sentence_transformer_predictions,
        ground_truth=np.ones_like(sentence_transformer_predictions),
    )
    log_performance(
        sentence_transformer_results,
        log_prefix=model_name.replace("/", "__"),
    )
    return dataset


def evaluate_ensemble_model(dataset: Dataset, column_names: list[str]):
    """Evaluate and log performance of ensemble model. The prediction of the ensemble "
    is simply the majority of the three columns in the dataset."""

    logger.info("Computing ensemble prediction")
    all_predictions = np.array([dataset[c] for c in column_names])
    ensemble_predictions = (all_predictions.sum(axis=0) >= 2).astype(int)
    ensemble_results = evaluate_classification(
        ensemble_predictions, ground_truth=np.ones_like(ensemble_predictions)
    )
    log_performance(results=ensemble_results, log_prefix="ensemble")

    return
