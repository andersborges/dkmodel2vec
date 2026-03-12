#!/bin/bash

# Simple bash script to run multiple training experiments with different parameters
# Usage: ./run_finetune.sh

set -e  # Exit on error

echo "======================================"
echo "DK-LLM2Vec-Model2Vec tokenlearn finetuning pipeline"
echo "======================================"
echo "Starting time: $(date)"
echo ""

# test with 1000 examples
#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_sif0.0001_strip_upper_case_strip_exotic_focus_pca_stem_normalize_embeddings --data-path features/features_100000_max_length_800 --limit-samples 1000

# full try with default params
#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_sif0.0001_strip_upper_case_strip_exotic_focus_pca_stem_normalize_embeddings --data-path features/features_100000_max_length_800

# decrease learning rate
#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings --data-path features/features_100000_max_length_800 --lr 0.0001

#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_sif0.0001_strip_upper_case_strip_exotic_focus_pca_stem_normalize_embeddings --data-path features/features_100000_max_length_800 --lr 0.00001

# best performing model before finetune with low learning rate
#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_vocab200000_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings --data-path features/features_100000_max_length_800 --lr 0.0001

#python scripts/finetune.py --model2vec-model-name scripts/models/dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_strip_uncommon_focus_pca_stem_normalize_embeddings --data-path features/features_100000_max_length_800 --lr 0.0001


echo ""
echo "======================================"
echo "All experiments completed successfully!"
echo "Ending time: $(date)"
echo "======================================"