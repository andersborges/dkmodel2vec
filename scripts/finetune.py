import argparse
import logging
from pathlib import Path

from datasets import load_from_disk  # , Dataset
import numpy as np
import torch
import mlflow
from model2vec import StaticModel

# from model2vec.distill import distill
from sklearn.decomposition import PCA

from tokenlearn.losses import Loss
from tokenlearn.model import StaticModelForFineTuning
# from tokenlearn.utils import collect_means_and_texts, create_vocab

from dkmodel2vec.config import MLFLOW_TRACKING_URI
from dkmodel2vec.evaluator import evaluate_model
from dkmodel2vec.logging import setup_logging
from dkmodel2vec.data_loader import load_data, add_splits
from dkmodel2vec.config import RANDOM_STATE
from dkmodel2vec.constants import HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN
from dkmodel2vec.logging import log_memory_usage
from dkmodel2vec.utils import iterable_dimension_reduction

setup_logging()
logger = logging.getLogger(__name__)


_DEFAULT_BATCH_SIZE = 256
_DEFAULT_LEARNING_RATE = 1e-3

SCRIPT_DIR = Path(__file__).parent
PARENT_DIR = SCRIPT_DIR.parent


def main() -> None:
    """Main function to train and save a Model2Vec model using tokenlearn."""
    parser = argparse.ArgumentParser(description="Train a Model2Vec using tokenlearn.")
    group = parser.add_mutually_exclusive_group(required=True)
    # group.add_argument(
    #     "--model-name",
    #     type=str,
    #     default=None,
    #     help="The model name for distillation (e.g., 'baai/bge-base-en-v1.5').",
    # )
    group.add_argument(
        "--model2vec-model-name",
        type=str,
        default=None,
        help="The Model2Vec model name or path to initialize from (e.g., 'vocquant').",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/fineweb_bgebase",
        help="Path to the directory containing the dataset.",
    )
    # parser.add_argument(
    #     "--save-path",
    #     type=str,
    #     required=True,
    #     help="Path to save the trained model.",
    # )
    # parser.add_argument(
    #     "--device",
    #     type=str,
    #     default="cuda",
    #     help="Device to run the training on (e.g., 'cpu', 'cuda').",
    # )
    # parser.add_argument(
    #     "--vocab-size",
    #     type=int,
    #     default=56000,
    #     help="The vocabulary size to use for training.",
    # )
    # parser.add_argument(
    #     "--trust-remote-code",
    #     action="store_true",
    #     help="Trust remote code when loading the model.",
    # )
    # parser.add_argument(
    #     "--pca-dims",
    #     type=int,
    #     default=256,
    #     help="Number of dimensions to reduce the target embeddings to using PCA.",
    # )
    # parser.add_argument("--use-wandb", action="store_true", help="Use Weights & Biases for logging.")
    # parser.add_argument("--project-name", type=str, default="tokenlearn", help="Weights & Biases project name.")
    #    parser.add_argument("--run-name", type=str, help="MLFlow run name")
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Limit the number of samples to use for training.",
    )
    parser.add_argument(
        "--loss",
        default="contrastive",
        choices=Loss.__members__.values(),
        help="The loss function to use for training.",
    )
    parser.add_argument(
        "--lr",
        default=_DEFAULT_LEARNING_RATE,
        type=float,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=_DEFAULT_BATCH_SIZE,
        help="Batch size for training.",
    )

    args = parser.parse_args()

    ds = load_from_disk(args.data_path)
    logger.info("Filtering on splits...")
    ds = ds.filter(lambda example: example["split"] in ["train", "val"], num_proc=4)
    log_memory_usage("After filtering on splits.")
    limit = args.limit_samples
    if limit:
        ds = ds.select(range(min(limit, ds.num_rows)))

    log_memory_usage("Extracting data from dataset. ")

    logger.info("Loading model2vec model...")
    model = StaticModel.from_pretrained(
        path=args.model2vec_model_name,
        quantize_to="float32",
    )

    #    in_dim = train_vec.shape[-1]
    out_dim = model.embedding.shape[-1]
    log_memory_usage("Reducing dimensions of target vectors...")

    ds = iterable_dimension_reduction(ds=ds, n_components=out_dim, batch_size=50_000)

    log_memory_usage("Reducing dimensions of target vectors...")

    loss = Loss(args.loss)

    run_name = (
        f"{args.model2vec_model_name.replace('/', '_')}-{args.data_path.split('/')[-1]}"
    )
    if limit is not None:
        run_name += f"-limit{limit}"
    if args.lr != _DEFAULT_LEARNING_RATE:
        run_name += f"-lr{args.lr}"

    logger.info("Creating trainable model...")
    trainable = StaticModelForFineTuning.from_static_model(
        model=model, out_dim=out_dim, loss=loss
    )

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment("finetuning")
    logger.info(f"Columns in dataset are {ds.column_names}")
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("batch_size", args.batch_size)
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("examples", ds.num_rows)
        logger.info("Fitting model...")
        trainable.fit(
            X=ds["document"],
            y=torch.from_numpy(np.array(ds["embedding"])),
            batch_size=args.batch_size,
            #            device=args.device,
            #            use_wandb=args.use_wandb,
            #            project_name=args.project_name,
            #            run_name=run_name,
            learning_rate=args.lr,
        )
        logger.info("Converting to static model...")
        static_model = trainable.to_static_model()

        finetune_dir = PARENT_DIR / "finetunes"
        finetune_dir.mkdir(parents=True, exist_ok=True)
        save_dir = finetune_dir / run_name
        logger.info("Saving fine-tuned model...")
        static_model.save_pretrained(str(save_dir))

        # assess performance
        logger.info("Loading data and model...")
        dsdk = load_data()
        dsdk = add_splits(dsdk)
        ds_test = dsdk.filter(lambda example: example["split"] == "test")
        ds_test_for_eval = ds_test.filter(
            lambda example: example[HAS_POSITIVE_AND_NEGATIVE_EXAMPLE_COLUMN]
        )
        ds_test_for_eval = evaluate_model(
            dataset=ds_test_for_eval,
            model=static_model,
            instruction_model=static_model,
        )


if __name__ == "__main__":
    main()
