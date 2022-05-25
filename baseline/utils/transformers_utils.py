class __Imports:
    import re
    import transformers
    from typing import Optional, List

LF_TOKENS = ("\n","\r","\t")

def from_pretrained_tokenizer(tokenizer_name_or_path:str, add_lf_tokens:bool=True, additional_special_tokens:__Imports.Optional[__Imports.List[str]]=None, trim_offsets:bool=False, force_deberta_v2_fast:bool=False) -> __Imports.transformers.PreTrainedTokenizer:
    added_special_tokens = list()
    if add_lf_tokens:
        added_special_tokens.extend(list(LF_TOKENS))
    if additional_special_tokens is not None:
        added_special_tokens.extend(list(additional_special_tokens))

    if force_deberta_v2_fast and (__Imports.re.search("deberta-v[23]", tokenizer_name_or_path)):
        tokenizer = __Imports.transformers.models.deberta_v2.DebertaV2TokenizerFast.from_pretrained(tokenizer_name_or_path, trim_offsets=trim_offsets)
    else:
        tokenizer = __Imports.transformers.AutoTokenizer.from_pretrained(tokenizer_name_or_path, trim_offsets=trim_offsets)

    if len(added_special_tokens) > 0:
        num_added_tokens = tokenizer.add_special_tokens({"additional_special_tokens":added_special_tokens})
    else:
        num_added_tokens = 0
    return tokenizer, num_added_tokens, added_special_tokens
