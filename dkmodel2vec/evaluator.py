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
import json
from typing import Dict, Tuple
from datasets import DatasetDict, Dataset
from rank_bm25 import BM25Okapi
from tokenizers import Tokenizer
from sentence_transformers import SentenceTransformer
from dkmodel2vec.config import BEST_SENTENCE_TRANSFORMER


def get_best_sentence_transformer():
    model = SentenceTransformer(BEST_SENTENCE_TRANSFORMER)
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
            example["bm25_prediction"] = -1
            return example
            

    # somehow BM25 does not work with only 2 examples? so adding a third
    corpus.append(["dummy", "document", "for", "idf", "calculation"])

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(q_tokens)[:2]  # get rid of the dummy document
    example["bm25_prediction"] = np.argmax(scores)

    return example

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
    positives = dataset["positive"]
    negatives = dataset["negative"]
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
            mlflow.log_metric(f"{log_prefix}_metric", results[metric])

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
    dataset: Dataset,
    model: StaticModel,
    instruction_model: StaticModel,
    max_samples: int | None = None,
) -> Dict:
    """Run the complete evaluation on the subset of the test set that contains both positive and negative examples."""
    dataset = dataset.filter(
        lambda example: True if example["has_positive_and_negative"] else False
    )
    if max_samples is not None:
        dataset = dataset.select(range(max_samples))

    ##### LLM2VEC2MODEL2VEC without instruction
    # Compute similarities
    print("Computing similarities with raw documents")
    pos_dists, neg_dists = compute_distances(encoder=model, dataset=dataset)

    print("Evaluating classification performance...")
    # Classification: positive class if pos_distance < neg_distance
    predictions = (pos_dists < neg_dists).astype(int)
    results = evaluate_classification(
        predictions, ground_truth=np.ones_like(predictions)
    )
    log_performance(results)

    ##### LLM2VEC2MODEL2VEC WITH instructions

    print("Computing similarities with documents using instructions...")
    pos_dists, neg_dists = compute_distances(
        encoder=model, instruction_encoder=instruction_model, dataset=dataset
    )

    print("Evaluating classification performance...")
    # Classification: positive class if pos_distance < neg_distance
    predictions_with_instruct = (pos_dists < neg_dists).astype(int)
    results = evaluate_classification(
        predictions_with_instruct, ground_truth=np.ones_like(predictions_with_instruct)
    )
    log_performance(results, log_prefix="instruct")

    #### BM25 performance
    bm25_results = evaluate_classification(
        dataset["bm25_prediction"], ground_truth=np.ones_like(predictions)
    )
    log_performance(bm25_results, log_prefix="bm25")

    #### Good sentence transformer for comparison
    print("Computing scores with good sentence transformer model... ")
    best_sentence_transformer = get_best_sentence_transformer()
    pos_dists, neg_dists = compute_distances(
        encoder=best_sentence_transformer,
        dataset=dataset,
        query_column_name="query_instruct",
    )
    sentence_transformer_predictions = (pos_dists < neg_dists).astype(int)
    log_performance(sentence_transformer_predictions, prefix=BEST_SENTENCE_TRANSFORMER)

    #### ENSEMBLE PREDICTION
    print("Computing ensemble prediction")
    all_predictions = np.array(
        [predictions, dataset["bm25_prediction"], sentence_transformer_predictions]
    )
    ensemble_predictions = (all_predictions.sum(axis=0) >= 2).astype(int)
    log_performance(ensemble_predictions, prefix="ensemble")

    return
