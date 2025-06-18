from dkmodel2vec.llm_loader import load_llm2vec_model

# from model2vec.distill import distill_from_model


if __name__ == "__main__": 
    model = load_llm2vec_model()

# Assuming a loaded model and tokenizer

#    m2v_model = distill_from_model(model=model, tokenizer=tokenizer, pca_dims=256)

 #   m2v_model.save_pretrained("m2v_model")