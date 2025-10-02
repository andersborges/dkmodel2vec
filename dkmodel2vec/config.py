VOCAB_SIZE = 150_000
DEFAULT_PATTERN = r"\[unused\d+\]"
WORD_CONTAINS_UPPER_CASE_PATTERN = r"\b\w*[A-Z]\w*\b"
CONTAINS_EXOTIC_PATTERN = r'^(?!Ġ[a-zA-ZæøåÆØÅ0-9.,\s]*$)(?!<\|end_of_text\|>$).*[^a-zA-ZæøåÆØÅ0-9.,\s]'
DANISH_INSTRUCTION = "Givet et spørgsmål, find relevante tekstudsnit, der besvarer det:"
E5_EMBED_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
BEST_SENTENCE_TRANSFORMER = "intfloat/multilingual-e5-large-instruct"
REFERENCE_MODEL2VEC = "minishlab/potion-base-8M"
FALLBACK_UNK_TOKEN = ","  # only active if no unk_token is found
SIF_COEFFICIENT = 1e-3
N_SPLITS = 10
RANDOM_STATE = 51
NORMALIZE_EMBEDDINGS = False # settings for final model2vec model, you can also change setting in model 
FOCUS_PCA = False
STEM = False
