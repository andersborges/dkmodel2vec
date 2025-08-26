# dkmodel2vec
This repo contains the code to train a static embedder from an LLM2Vec model for the Danish language and assess its performance. 

This is useful if need ultrafast embeddings that are  tailored to your own data and you are willing to sacrifice on performance compared to a big shiny model. 

The code heavily relies on the packages model2vec and LLM2vec. 

## Features

- Train model2vec (static) model from big fance LLM2Vec model
- Allows training of seperate Instruction and Retrieval encodings
- Compare performance to simple BM25 encoder and a 'good' Sentence Transformer

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh
````

```bash
# Install packages
uv sync
```

```bash
# Install the local package
uv pip install -e . 
```
## Usage
To train the model you will probably want a GPU to perform inference on the many tokens. 

First start the MLflow server
```bash
uv run sh scripts/run_mlflow.sh
```

With the server running run the training the script: 
```bash 
uv run python scripts/train.py
```

### Advanced Usage
You can modify the training script to work for your own data and LLM2Vec model. Look in the the ```dkmodel2vec.load_data.py``` and ```dkmodel2vec.data_loader.py``` files. 

## Requirements
- Python >3.12 (UV will install a suitable version)
- You might also need a huggingface token to download the datasets. 

