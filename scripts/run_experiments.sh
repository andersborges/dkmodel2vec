#!/bin/bash

# Simple bash script to run multiple training experiments with different parameters
# Usage: ./run_experiments.sh

set -e  # Exit on error

echo "======================================"
echo "DK-LLM2Vec-Model2Vec Training Pipeline"
echo "======================================"
echo "Starting time: $(date)"
echo ""

# # Run baselines
# python scripts/baselines.py

# # Run experiments with pre-normalization
# python scripts/hyperparams.py --output-dim 256 --prenormalize-embeddings
# python scripts/hyperparams.py --output-dim 256 --pre-pca-normalize-embeddings
# python scripts/hyperparams.py --output-dim 256 --prenormalize-embeddings --pre-pca-normalize-embeddings

# # Run experiments with different output dimensions (default vocab and sif)
# python scripts/hyperparams.py --output-dim 256
# python scripts/hyperparams.py --output-dim 512
# python scripts/hyperparams.py --output-dim 1024
# python scripts/hyperparams.py --output-dim 2048
# python scripts/hyperparams.py --output-dim 4096
#python scripts/hyperparams.py --output-dim none


# Run experiments with different SIF coefficients
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.01
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.005
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.001
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0005
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0001

# Run experiments with different vocabulary sizes
#python scripts/hyperparams.py --output-dim 256 --vocab-size 100000
#python scripts/hyperparams.py --output-dim 256 --vocab-size 150000
#python scripts/hyperparams.py --output-dim 256 --vocab-size 200000

# Remove lower-case tokens from output tokenizer vocab
#python scripts/hyperparams.py --output-dim 256 --strip-upper-case

# Remove both lower-case tokens and strip exotic tokens from tokenizer vocab
#python scripts/hyperparams.py --output-dim 256 --strip-upper-case --strip-exotic

# Focus dimension reduction (PCA) on the embeddings space that is represented in the corpus.
#python scripts/hyperparams.py --output-dim 256 --strip-upper-case --strip-exotic --focus-pca


# Normalize aggregated embeddings
#python scripts/hyperparams.py --output-dim 256 --normalize-embeddings

# Ignore all external tokens in training
#python scripts/hyperparams.py --output-dim 256 --ignore-external-tokens

# Normalized and different values of SIF
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.01 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 200000 
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 200000
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.001 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 200000

# # Normalized and different values of SIF but where vocab size is 150k to avoid running out of memory
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0001 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.01 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000 
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.001 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000

# Effect of stemming before creating vocabulary
#python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --stem --vocab-size 150000 
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0001 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --stem --vocab-size 150000 
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.001 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --stem --vocab-size 150000 
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --stem --vocab-size 150000 


# # train on full dataset
# python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000 --full-dataset

# # train on full dataset with stem
python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.0005 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000 --full-dataset --stem


echo ""
echo "======================================"
echo "All experiments completed successfully!"
echo "Ending time: $(date)"
echo "======================================"