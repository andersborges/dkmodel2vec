class LlamaModelWrapper:
    """Wrapper to make LlamaModel compatible with model2vec distillation."""

    def __init__(self, model):
        # If it's an LLM2Vec, extract the inner LlamaModel
        if hasattr(model, "model"):
            self._model = model.model  # This is the actual LlamaModel
        else:
            self._model = model

    def __call__(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        # Standard transformer call - ignore token_type_ids
        return self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)

    def to(self, *args, **kwargs):
        self._model = self._model.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self._model.parameters()).device

    def __getattr__(self, name):
        return getattr(self._model, name)
