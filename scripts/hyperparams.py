#!/usr/bin/env python
"""
Training script for dk-llm2vec-model2vec with configurable parameters.
"""
from typing import Union, Literal

import argparse
import logging
import mlflow
import torch
import gc
from pathlib import Path
from copy import deepcopy
from sklearn.model_selection import train_test_split
import numpy as np
from model2vec.distill.inference import PCADimType

from dkmodel2vec.llm_loader import load_llm2vec_model
from dkmodel2vec.data_loader import load_data
from dkmodel2vec.config import (
    VOCAB_SIZE,
    DEFAULT_PATTERN, 
    WORD_CONTAINS_UPPER_CASE_PATTERN, 
    CONTAINS_EXOTIC_PATTERN, 
    SIF_COEFFICIENT,
    RANDOM_STATE, 
    NORMALIZE_EMBEDDINGS, 
    FOCUS_PCA, 
    STEM
)
from dkmodel2vec.constants import (
    HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN,
    DATASET_NEGATIVE_COLUMN,
    DATASET_POSITIVE_COLUMN,
    DATASET_QUERY_COLUMN,
)
from dkmodel2vec.evaluator import evaluate_model
from dkmodel2vec.vocab import create_vocabulary, lower_case_tokenizer
from dkmodel2vec.distillation import distill_from_llm2vec_and_corpus
from dkmodel2vec.logging import setup_logging

setup_logging()
logger = logging.getLogger(__name__)


def cleanup_memory():
    """Clean up memory after training iteration."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def log_gpu_memory():
    """Log GPU memory usage if available."""
    if torch.cuda.is_available():
        logger.info(f"GPU memory allocated: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
        logger.info(f"GPU memory reserved: {torch.cuda.memory_reserved()/1024**3:.2f} GB")


def generate_run_name(
    output_dim: int,
    vocab_size: int = VOCAB_SIZE,
    sif_coefficient: float = SIF_COEFFICIENT,
    test_size: float = 0.1,
    random_state: int = RANDOM_STATE,
    full_dataset_training: bool = False,
    evaluate: bool = True,
    prenormalize_embeddings: bool = False,
    prePCAnormalize_embeddings: bool = False,
    focus_pca: bool = False, 
    stem: bool = False,
    strip_upper_case: bool = False, 
    strip_exotic: bool = False, 
    normalize_embeddings: bool = NORMALIZE_EMBEDDINGS, 
    ignore_external_tokens: bool = False, 
):
    """Generate run name based on non-default parameters."""
    # Define defaults for comparison
    defaults = {
        'vocab_size': VOCAB_SIZE,
        'sif_coefficient': SIF_COEFFICIENT,
        'test_size': 0.1,
        'random_state': RANDOM_STATE,
        'full_dataset_training': False,
        'evaluate': True,
        'prenormalize_embeddings': False, 
        'prePCAnormalize_embeddings': False, 
        'focus_pca': FOCUS_PCA,
        'stem' : STEM, 
        'strip_upper_case' : False, 
        'strip_exotic': False, 
        'normalize_embeddings' : NORMALIZE_EMBEDDINGS, 
        'ignore_external_tokens' : False
    }
    
    # Always include output_dim as it's required
    params = [f"dim{output_dim}"]
    
    # Check for non-default values
    if vocab_size != defaults['vocab_size']:
        params.append(f"vocab{vocab_size}")
    if sif_coefficient != defaults['sif_coefficient']:
        params.append(f"sif{sif_coefficient}")
    if test_size != defaults['test_size']:
        params.append(f"test{test_size}")
    if random_state != defaults['random_state']:
        params.append(f"seed{random_state}")
    if full_dataset_training != defaults['full_dataset_training']:
        params.append("full")
    if evaluate != defaults['evaluate']:
        params.append("noeval")
    if strip_upper_case != defaults['strip_upper_case']:
        params.append("strip_upper_case")
    if strip_exotic != defaults['strip_exotic']:
        params.append("strip_exotic")
    if prenormalize_embeddings != defaults['prenormalize_embeddings']:
        params.append("prenormalize_embeds")
    if prePCAnormalize_embeddings != defaults['prePCAnormalize_embeddings']:
        params.append("prePCAnormalize_embeds")
    if focus_pca != defaults['focus_pca']:
        params.append("focus_pca")
    if stem != defaults['stem']:
        params.append("stem")
    if normalize_embeddings!= defaults['normalize_embeddings']:
        params.append("normalize_embeddings")
    if ignore_external_tokens!= defaults['ignore_external_tokens']:
        params.append("ignore_external_tokens")
    
    return "_".join(params)


def train_model(
    output_dim:PCADimType,
    vocab_size: int = VOCAB_SIZE,
    sif_coefficient: float = SIF_COEFFICIENT,
    test_size: float = 0.1,
    random_state: int = RANDOM_STATE,
    mlflow_tracking_uri: str = "http://localhost:8000",
    mlflow_experiment: str = "llm2model2vec",
    run_name: str = None,
    save_model: bool = True,
    strip_upper_case: bool = False, 
    strip_exotic: bool = False, 
    models_dir: str = None,
    full_dataset_training: bool = False,
    evaluate: bool = True,
    prenormalize_embeddings: bool = False,
    prePCAnormalize_embeddings: bool = False, 
    focus_pca: bool = FOCUS_PCA,
    stem: bool = STEM,
    normalize_embeddings: bool = NORMALIZE_EMBEDDINGS, 
    ignore_external_tokens: bool = False
):
    """
    Train a single model with specified parameters.
    
    Args:
        output_dim: Output dimension for PCA
        vocab_size: Size of vocabulary
        sif_coefficient: SIF coefficient for weighting
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        mlflow_tracking_uri: MLflow tracking server URI
        mlflow_experiment: MLflow experiment name
        run_name: Name for MLflow run
        save_model: Whether to save the trained model
        strip_upper_case: Whether to strip tokens containing upper case 
        strip_exotic: Whether to strip tokens that contain exotic characters. 
        models_dir: Directory to save models
        full_dataset_training: Whether to train on full dataset (no train/test split)
        evaluate: Whether to evaluate the model
        evaluate_baselines: Whether to evaluate baseline models
        prenormalize_embeddings: Whether to normalize embeddings of tokens before weighting them
        prePCAnormalize_embeddings: Whether to normalize embeddings of tokens before PCA
        focus_pca: Whether to focus the pca on the token embeddings that are represented in the corpus. 
        stem: Whether to use a Danish stemmer on vocabulary.
        normalize_embeddings: Whether the aggregated embeddings should be normalized. 
        ignore_external_tokens: Whether to ignore all external tokens from vocabulary and ignore their weighting
    """
    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_tracking_uri)
    mlflow.set_experiment(mlflow_experiment)
    
    # Load data and model
    logger.info("Loading data and model...")
    dsdk = load_data()
    model = load_llm2vec_model()
    tokenizer = lower_case_tokenizer(model.tokenizer)
        
    # Generate run name based on non-default parameters
    if run_name is None:
        run_name = generate_run_name(
            output_dim=output_dim,
            vocab_size=vocab_size,
            sif_coefficient=sif_coefficient,
            test_size=test_size,
            random_state=random_state,
            full_dataset_training=full_dataset_training,
            evaluate=evaluate,
            strip_upper_case=strip_upper_case,
            strip_exotic=strip_exotic,
            prenormalize_embeddings=prenormalize_embeddings,
            prePCAnormalize_embeddings=prePCAnormalize_embeddings, 
            focus_pca=focus_pca,
            stem = stem,
            normalize_embeddings=normalize_embeddings, 
            ignore_external_tokens=ignore_external_tokens
        )
    
    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param("output_dim", output_dim)
        mlflow.log_param("vocab_size", vocab_size)
        mlflow.log_param("sif_coefficient", sif_coefficient)
        mlflow.log_param("dataset_size", dsdk.num_rows)
        mlflow.log_param("weight_of_unseen_tokens", "min")
        mlflow.log_param("full_dataset_training", full_dataset_training)
        mlflow.log_param("test_size", test_size if not full_dataset_training else 0)
        mlflow.log_param("random_state", random_state)
        mlflow.log_param("prenormalize_embeddings", prenormalize_embeddings)
        mlflow.log_param("prePCAnormalize_embeddings", prePCAnormalize_embeddings)
        mlflow.log_param("strip_upper_case", strip_upper_case)
        mlflow.log_param("strip_exotic", strip_exotic)
        mlflow.log_param("normalize_embeddings", normalize_embeddings)
        mlflow.log_param("ignore_external_tokens", ignore_external_tokens)
        mlflow.log_param("focus_pca", focus_pca)
        mlflow.log_param("stem", stem)
        if full_dataset_training:
            # Use full dataset for training
            logger.info("Training on full dataset...")
            texts = (
                list(dsdk[DATASET_QUERY_COLUMN])
                + list(dsdk[DATASET_POSITIVE_COLUMN])
                + list(dsdk[DATASET_NEGATIVE_COLUMN])
            )
            ds_test_for_eval = dsdk.filter(
                lambda example: example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN]
            )
        else:
            # Split data
            logger.info(f"Splitting data with test_size={test_size}...")
            train_idx, test_idx = train_test_split(
                np.arange(dsdk.num_rows),
                test_size=test_size,
                stratify=dsdk["has_positive_and_negative"],
                random_state=random_state,
                shuffle=True
            )
            ds_train, ds_test = dsdk.select(train_idx), dsdk.select(test_idx)
            
            # Get training texts
            texts = (
                list(ds_train[DATASET_QUERY_COLUMN])
                + list(ds_train[DATASET_POSITIVE_COLUMN])
                + list(ds_train[DATASET_NEGATIVE_COLUMN])
            )
            
            # Prepare test set for evaluation
            ds_test_for_eval = ds_test.filter(
                lambda example: example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN]
            )
        token_remove_pattern = DEFAULT_PATTERN
        if strip_upper_case:
            word_contains_upper_case_pattern = WORD_CONTAINS_UPPER_CASE_PATTERN
            token_remove_pattern = r"|".join([token_remove_pattern,word_contains_upper_case_pattern] )
        if strip_exotic: 
            contains_exotic_pattern = CONTAINS_EXOTIC_PATTERN
            token_remove_pattern = r"|".join([token_remove_pattern, contains_exotic_pattern])

        # Create vocabulary
        if not ignore_external_tokens:
            logger.info(f"Creating vocabulary with size {vocab_size}...")
            vocabulary = create_vocabulary(
                texts, 
                vocab_size=vocab_size,
                stem=stem
                )
        else: 
            vocabulary = None       
        # Train model
        logger.info(f"Training model with output_dim={output_dim}...")
        m2v_model = distill_from_llm2vec_and_corpus(
            model=model,
            tokenizer=tokenizer,
            vocabulary=vocabulary,
            corpus=texts,
            pca_dims=output_dim,
            apply_zipf=True,
            sif_coefficient=sif_coefficient,
            token_remove_pattern=token_remove_pattern,
            prenormalize_embeddings=prenormalize_embeddings,
            prePCAnormalize_embeddings=prePCAnormalize_embeddings, 
            focus_pca=focus_pca,
            normalize_embeddings = normalize_embeddings
        )
        ds_test_for_eval = evaluate_model(
                dataset=ds_test_for_eval,
                model=m2v_model,
                instruction_model=m2v_model,
            )


        # Save model
        if save_model:
            if models_dir is None:
                models_dir = Path(__file__).parent / "models"
            else:
                models_dir = Path(models_dir)
            
            models_dir.mkdir(exist_ok=True, parents=True)
            
            model_name = f"dk-llm2vec-model2vec-{run_name}"
            save_path = models_dir / model_name
            
            logger.info(f"Saving model to {save_path}...")
            m2v_model.save_pretrained(save_path)
            
            # Log the model path
            mlflow.log_param("model_save_path", str(save_path))
#            mlflow.log_artifact(str(save_path))
        
        # Log memory usage
        log_gpu_memory()
        
        # Cleanup
        logger.info("Cleaning up memory...")
        del m2v_model
        if evaluate:
            del ds_test_for_eval
        cleanup_memory()
        log_gpu_memory()
        
        logger.info(f"Training completed for output_dim={output_dim}")


def parse_pca_dim(value: str) -> PCADimType:
    if value.lower() == "none":
        return None
    elif value.lower() == "auto":
        return "auto"
    else:
        try:
            return int(value)
        except ValueError:
            raise argparse.ArgumentTypeError(f"Invalid PCA dimension: {value}")

def main():
    parser = argparse.ArgumentParser(
        description="Train dk-llm2vec-model2vec with configurable parameters"
    )

    # Required arguments
    parser.add_argument(
        "--output-dim",
        type=parse_pca_dim,
        required=True,
        help="Output dimension for PCA"
    )
    
    # Optional arguments
    parser.add_argument(
        "--strip-upper-case",
        action="store_true",
        help="Strip tokens containing uppercase letters (default: False)"
    )
    parser.add_argument(
        "--strip-exotic",
        action="store_true",
        help="Strip tokens containing exotic characters (default: False)"
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=VOCAB_SIZE,
        help=f"Vocabulary size (default: {VOCAB_SIZE})"
    )
    parser.add_argument(
        "--sif-coefficient",
        type=float,
        default=SIF_COEFFICIENT,
        help=f"SIF coefficient (default: {SIF_COEFFICIENT})"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.1,
        help="Test set size as fraction (default: 0.1)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=RANDOM_STATE,
        help="Random seed (default: 51)"
    )
    parser.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default="http://localhost:8000",
        help="MLflow tracking URI (default: http://localhost:8000)"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="llm2model2vec",
        help="MLflow experiment name (default: llm2model2vec)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (default: auto-generated)"
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default=None,
        help="Directory to save models (default: ./models)"
    )
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Train on full dataset without train/test split"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save the trained model"
    )
    parser.add_argument(
        "--no-eval",
        action="store_true",
        help="Skip evaluation"
    )
    
    parser.add_argument(
        "--prenormalize-embeddings",
        action="store_true",
        help="Prenormalize embeddings before weighting (default: False)"
    )
    parser.add_argument(
        "--normalize-embeddings",
        action="store_true",
        help="Normalize aggregated embeddings at inference (default: False)"
    )
    parser.add_argument(
        "--ignore-external-tokens",
        action="store_true",
        help="Ignore external tokens and only use the internal vocabulary (default: False)"
    )
    parser.add_argument(
        "--focus-pca",
        action="store_true",
        help="Focus the dimension reduction (PCA) to the embeddings represented by tokens in the corpus. (default: False)"
    )
    parser.add_argument(
        "--stem",
        action="store_true",
        help="Use Danish stemmer. (default: False)"
    )

    parser.add_argument(
        "--pre-pca-normalize-embeddings",  # This becomes args.pre_pca_normalize_embeddings
        action="store_true",
        help="Prenormalize embeddings before PCA (default: False)"
    )    
    args = parser.parse_args()
    
    # Train the model
    train_model(
        output_dim=args.output_dim,
        vocab_size=args.vocab_size,
        sif_coefficient=args.sif_coefficient,
        test_size=args.test_size,
        random_state=args.random_state,
        mlflow_tracking_uri=args.mlflow_tracking_uri,
        mlflow_experiment=args.mlflow_experiment,
        run_name=args.run_name,
        save_model=not args.no_save,
        strip_upper_case=args.strip_upper_case,
        strip_exotic=args.strip_exotic,
        models_dir=args.models_dir,
        full_dataset_training=args.full_dataset,
        evaluate=not args.no_eval,
        prenormalize_embeddings=args.prenormalize_embeddings,
        prePCAnormalize_embeddings=args.pre_pca_normalize_embeddings,
        focus_pca=args.focus_pca,
        stem=args.stem,
        normalize_embeddings=args.normalize_embeddings,  
        ignore_external_tokens=args.ignore_external_tokens
        )


if __name__ == "__main__":
    main()