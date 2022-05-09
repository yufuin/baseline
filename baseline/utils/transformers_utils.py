class __Imports:
    import transformers

def from_pretrained_tokenizer(tokenizer_name_or_path:str, add_lf_tokens:bool=False, trim_offsets:bool=False) -> __Imports.transformers.PreTrainedTokenizer:
    num_added_tokens = 0
    tokenizer = __Imports.transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, trim_offsets=trim_offsets)
    if add_lf_tokens:
        num_added_tokens += tokenizer.add_special_tokens({"additional_special_tokens":["\n","\r","\t"]})
    return tokenizer, num_added_tokens
