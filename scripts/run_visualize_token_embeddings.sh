#!/bin/bash

# Simple bash script to run dumping of dimension-reduced token embeddings

set -e  # Exit on error

echo "======================================"
echo "Dumping token embeddings reduced to 2D for easy visualization."
echo "======================================"
echo "Starting time: $(date)"
echo ""

# basic comparison
python scripts/visualize_token_embeddings.py --model-path scripts/models/dk-llm2vec-model2vec-dim256_ignore_external_tokens --outpath data/raw.parquet --other-model-path scripts/models/dk-llm2vec-model2vec-dim256_strip_upper_case_strip_exotic_strip_uncommon



echo ""
echo "======================================"
echo "Script completed successfully!"
echo "Ending time: $(date)"
echo "======================================"