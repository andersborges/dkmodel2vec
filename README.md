# dkmodel2vec
This repo contains the code to train a [static embedder from an LLM2Vec](https://huggingface.co/andersborges/model2vecdk) model for the Danish language and assess its performance. 

This is useful if need ultrafast embeddings that are  tailored to your own data and you are willing to sacrifice on performance compared to a big shiny model.

The code heavily relies on the packages model2vec, tokenlearn and LLM2vec. 

## Features

- Train model2vec (static) model from big fancy LLM2Vec model
- Allows training of seperate Instruction and Retrieval encodings
- Compare performance to simple BM25 encoder and a 'good' Sentence Transformer
- Allows finetuning with tokenlearn with query, positive and negative examples
- Assess performance of retriever, hybrid retrieval with BM25 and two-stage hybrid retrieval with a reranker
- Visualize embeddings as sum of contributions

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

# On machines with GPU
uv pip install -e ".[gpu]"

# On CPU-only machines
uv pip install -e ".[cpu]"
```
## Usage
To train the model you will probably want a GPU to perform inference on the many tokens. 

First start the MLflow server
```bash
uv run sh scripts/run_mlflow.sh
```

The UI will be available at localhost:8000. 

### Training a model

With the server running you can train a model with a set of hyperparameters with e.g.: 
```bash 
uv run python scripts/hyperparams.py --output-dim 256 --sif-coefficient 0.01 --strip-upper-case --strip-exotic --focus-pca --normalize-embeddings --vocab-size 150000 
```

### Finetuning a model
Tokenlearn allows finetuning of the trained model but requires a dataset. 

You can generate it with e.g.: 
```bash 
uv run python scripts/featurize.py --max-means 1000000 --max-length 800 --batch_size 32
```

To run the training run: 
```bash 
uv run python scripts/finetune.py --model2vec-model-name <path-to-your-dumped-model> --data-path <path-to-your-dumped-dataset>
```

### Assessing use as a retriever
You can assess the retrieval performance of the model with: 

```bash 
uv run python scripts/retrieval.py --model <path-to-your-fine-tuned-model>
```

### Interpreting embeddings
```python 
m2v = StaticModel.from_pretrained("andersborges/model2vecdk")

# Create visualizer
viz = TokenEmbeddingVisualizer(model=m2v)
texts = ["Elefanten har et lang næse", "Elefanten bruger sin lange næse","Næsen på elefanten er stor"]
viz.visualize(texts)
```


### Advanced Usage
You can modify the training script to work for your own data and LLM2Vec model. Look in the the ```dkmodel2vec.load_data.py``` and ```dkmodel2vec.data_loader.py``` files. 

## Requirements
- Python 3.10 (UV will install a suitable version)
- You might also need a huggingface token to download the datasets. 

