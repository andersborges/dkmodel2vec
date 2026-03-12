#!/bin/bash

# Simple bash script to run multiple training experiments with different parameters
# Usage: ./run_retrieval.sh

set -e  # Exit on error

echo "======================================"
echo "DK-LLM2Vec-Model2Vec retrieval evaluation pipeline"
echo "======================================"
echo "Starting time: $(date)"
echo ""

# finetuned on stemmed model
#python scripts/retrieval.py --model finetunes/scripts_models_dk-llm2vec-model2vec-dim256_sif0.0001_strip_upper_case_strip_exotic_focus_pca_stem_normalize_embeddings-features_100000_max_length_800

# finetuned on non-stemmed model
#python scripts/retrieval.py --model finetunes/scripts_models_dk-llm2vec-model2vec-dim256_vocab200000_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings-features_100000_max_length_800

# best model?
#python scripts/retrieval.py --model finetunes/scripts_models_dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_focus_pca_normalize_embeddings-features_100000_max_length_800
	
# finetuned on stemmed model with more stripped tokens
#python scripts/retrieval.py --model finetunes/scripts_models_dk-llm2vec-model2vec-dim256_sif0.0005_strip_upper_case_strip_exotic_strip_uncommon_focus_pca_stem_normalize_embeddings-features_100000_max_length_800-lr0.0001

echo ""
echo "======================================"
echo "All experiments completed successfully!"
echo "Ending time: $(date)"
echo "======================================"