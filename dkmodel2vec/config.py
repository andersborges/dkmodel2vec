VOCAB_SIZE = 150_000
DANISH_INSTRUCTION = "Givet et spørgsmål, find relevante tekstudsnit, der besvarer det:"
E5_EMBED_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query"
)
BEST_SENTENCE_TRANSFORMER = "intfloat/multilingual-e5-large-instruct"
REFERENCE_MODEL2VEC = "minishlab/potion-base-8M"
FALLBACK_UNK_TOKEN = ","  # only active if no unk_token is found
SIF_COEFFICIENT = 1e-3
N_SPLITS = 10
