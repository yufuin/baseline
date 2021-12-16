class SubwordOffsetTokenizer:
    """
    NOTE: This is no longer needed since official huggingface/transformers provides an option to have the offsets (see Tokenizer.return_offsets_mapping).
    """
    def __init__(self, tokenizer):
        """
        tokenizer : transformers.AutoTokenizer (tested on transformers==2.5.0)
        """
        self.tokenizer = tokenizer
        self.word_to_subword_ids = dict()

    def _get_subword_ids(self, word):
        assert type(word) is str
        if word not in self.word_to_subword_ids:
            self.word_to_subword_ids[word] = self.tokenizer.encode(word, add_special_tokens=False, add_prefix_space=True)
        return self.word_to_subword_ids[word]

    def encode(self, words, add_special_tokens=True):
        assert type(words) in [list, tuple]
        subword_offsets = list() # subword_offsets[word_index] == subword_head_index
        subword_ids_without_special_tokens = list()
        for word in words:
            subword_offsets.append(len(subword_ids_without_special_tokens))
            subword_ids_without_special_tokens.extend(self._get_subword_ids(word))

        if add_special_tokens:
            subword_ids = self.tokenizer.encode(subword_ids_without_special_tokens, add_special_tokens=add_special_tokens)
            len_subword_ids_without_special_tokens = len(subword_ids_without_special_tokens)
            if subword_ids_without_special_tokens != subword_ids[:len_subword_ids_without_special_tokens]:
                max_possible_shift = len(subword_ids) - len(subword_ids_without_special_tokens)
                for prefix_shift in range(1,max_possible_shift+1):
                    if subword_ids_without_special_tokens == subword_ids[prefix_shift:prefix_shift+len_subword_ids_without_special_tokens]:
                        break
                else:
                    raise ValueError("cannot align: {} -> {}".format(subword_ids, subword_ids_without_special_tokens))
                subword_offsets = [prefix_shift+o for o in subword_offsets]
        else:
            subword_ids = subword_ids_without_special_tokens
        return {"input_ids":subword_ids, "subword_offsets":subword_offsets}

    @classmethod
    def get_offset_matrix(cls, encoded):
        """
        encoded : output of SubwordOffsetTokenizer.encode
        output : 2D python nested-list ([len(subwords), len(words)]). output[subword_index][word_index] is 1.0 for corresponding offsets, otherwise 0.0.
        """
        assert ("input_ids" in encoded) and ("subword_offsets" in encoded)
        subword_offsets = encoded["subword_offsets"]
        word_seq_len =  len(subword_offsets)
        subword_seq_len = len(encoded["input_ids"])
        output = [[0.0 for _ in range(word_seq_len)] for _ in range(subword_seq_len)]
        for word_index,subword_index in enumerate(subword_offsets):
            output[subword_index][word_index] = 1.0
        return output
