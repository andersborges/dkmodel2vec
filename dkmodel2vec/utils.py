from transformers import AutoTokenizer

def check_fits_length(
    batch: dict[str, list], 
    tokenizer: AutoTokenizer, 
    max_length: int, 
    doc_max_length: int
) -> dict[str, list[bool]]:
    """Check if texts fit within doc_max_length without truncation."""
    texts = batch['document']
    
    tokenized = tokenizer(
        texts,
        padding=False,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    
    fits_length = [len(token_ids) <= doc_max_length for token_ids in tokenized["input_ids"]]
    
    return {'fits_length': fits_length}


def add_instruction_to_text(
    batch: dict[str, list], 
    query_instruction: str
) -> dict[str, list[str]]:
    """Add instruction prefix to texts. Assumes texts already fit length requirements."""
    texts = batch['document']
    columns = batch['column']
    
    processed_texts = []
    
    for text, column in zip(texts, columns):
        instruction = query_instruction if column == "query" else ""
        
        if instruction:
            processed_text = f"{instruction.strip()} !@#$%^&*(){text}"
        else:
            processed_text = f"!@#$%^&*(){text}"
        
        processed_texts.append(processed_text)
    
    return {'processed_text': processed_texts}