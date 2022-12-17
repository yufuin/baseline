# %%
import logging as _logging
_logger = _logging.getLogger(__name__)
_logger.setLevel(_logging.WARNING)
_ch = _logging.StreamHandler()
_ch.setLevel(_logging.WARNING)
_formatter = _logging.Formatter('%(name)s - %(levelname)s:%(message)s')
_ch.setFormatter(_formatter)
_logger.addHandler(_ch)

_TWO_BECAUSE_OF_SPECIAL_TOKEN = 2


import re as _re
import math as _math
import copy as _copy
import collections as _collections
import dataclasses as _D
import enum as _enum
from typing import List as _List, Optional as _Optional, Tuple as _Tuple, Union as _Union, Any as _Any

import numpy as _numpy
import transformers as _transformers

# %%
class TokenizerInterface:
    def __call__(self, *args, **kwargs) -> dict:
        raise NotImplementedError("TokenizerInterface.__call__")
    def build_inputs_with_special_tokens(self, *args, **kwargs) -> _List[int]:
        raise NotImplementedError("TokenizerInterface.build_inputs_with_special_tokens")

# %%
@_D.dataclass(frozen=True, eq=True)
class NERSpan:
    s: int
    e: int
    l: int
    id: _Any = None
    @classmethod
    def load_from_dict(cls, dumped):
        return cls(**dumped)
    @property
    def start(self):
        return self.s
    @property
    def end(self):
        return self.e
    @property
    def label(self):
        return self.l
    def check_some(self):
        assert isinstance(self.s, int)
        assert isinstance(self.e, int)
        assert self.s < self.e
        assert isinstance(self.l, int)
        return self
    def without_id(self):
        return _D.replace(self, id=None)

NERSpanAsList = _Union[_Tuple[int,int,int], _Tuple[int,int,int,_Any]]

class NERTaggingScheme(str, _enum.Enum):
    BILOU = "bilou"
    BIO = "bio"
    TOKEN_LEVEL = "token_level"

class NERLabelScheme(str, _enum.Enum):
    SINGLE_LABEL = "single_label"
    MULTI_LABEL = "multi_label"
    SPAN_ONLY = "span_only"

class NERSpanTag(int, _enum.Enum):
    O = 0
    B = 1
    I = 2
    L = 3
    U = 4

class NERTruncationScheme(str, _enum.Enum):
    NONE = "none"
    TRUNCATE = "truncate"
    SPLIT = "split"

@_D.dataclass
class NERInstance:
    text: str
    spans: _List[NERSpan]
    id: _Any = None

    input_ids: _Optional[_List[int]] = None
    offset_mapping_start: _Optional[_List[int]] = None
    offset_mapping_end: _Optional[_List[int]] = None
    is_added_special_tokens: bool = False
    token_spans: _Optional[_List[NERSpan]] = None

    metadata: dict = _D.field(default_factory=dict)
    note: _Any = None

    @classmethod
    def load_from_dict(cls, dumped):
        dumped = dict(dumped)
        dumped["spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["spans"]]
        if isinstance(dumped["token_spans"], list):
            dumped["token_spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["token_spans"]]
        return cls(**dumped)

    @classmethod
    def build(cls, text:str, spans:_List[_Union[NERSpan,NERSpanAsList]], id:_Any=None, *, check_some:bool=True, tokenizer:_Optional[TokenizerInterface]=None, add_special_tokens:_Optional[bool]=None, truncation:_Union[None, bool, NERTruncationScheme]=None, max_length:_Optional[int]=None, stride:_Optional[int]=None, fuzzy:_Optional[bool]=None, tokenizer_other_kwargs:_Optional[dict]=None):
        spans = [span if type(span) is NERSpan else NERSpan(*span) for span in spans]
        out = cls(text=text, spans=spans, id=id)
        if tokenizer is not None:
            encode_func_args = dict()
            for key in ["add_special_tokens", "truncation", "max_length", "stride", "fuzzy", "tokenizer_other_kwargs"]:
                value = eval(key)
                if value is not None:
                    encode_func_args[key] = value
            out = out.encode_(tokenizer, **encode_func_args)
        if check_some:
            if type(out) in [list, tuple]:
                [each_out.check_some() for each_out in out]
            else:
                out.check_some()
        return out

    def check_some(self):
        for span in self.spans:
            span.check_some()
        if self.token_spans is not None:
            for span in self.token_spans:
                span.check_some()
        return self

    def encode_(self, tokenizer:TokenizerInterface, *, add_special_tokens:bool=False, truncation:_Union[None, bool, NERTruncationScheme]=NERTruncationScheme.NONE, max_length:_Optional[int]=None, stride:_Optional[int]=None, fuzzy:bool=True, tokenizer_other_kwargs:_Optional[dict]=None):
        # reset
        self.is_added_special_tokens = False

        if isinstance(truncation, bool):
            if truncation:
                truncation = NERTruncationScheme.TRUNCATE
            else:
                truncation = NERTruncationScheme.NONE
        elif truncation is None:
            truncation = NERTruncationScheme.NONE

        if tokenizer_other_kwargs is None:
            tokenizer_other_kwargs = dict()
        else:
            tokenizer_other_kwargs = dict(tokenizer_other_kwargs)

        assert "truncation" not in tokenizer_other_kwargs, '"truncation" option can be only given at the direct argument of encode_ function.'
        for key in ["add_special_tokens", "max_length"]:
            if key in tokenizer_other_kwargs:
                _logger.warning(f'found the argument "{key}" in "tokenizer_other_kwargs". change to giving the argument directly to the fucntion.')
                exec(f'{key} = tokenizer_other_kwargs["{key}"]')
        tokenizer_other_kwargs["add_special_tokens"] = False
        if max_length is not None:
            tokenizer_other_kwargs["max_length"] = max_length

        if truncation == NERTruncationScheme.NONE:
            pass
        elif truncation == NERTruncationScheme.TRUNCATE:
            assert max_length is not None
            tokenizer_other_kwargs["truncation"] = True
            if add_special_tokens:
                max_length = max_length - _TWO_BECAUSE_OF_SPECIAL_TOKEN
                tokenizer_other_kwargs["max_length"] = tokenizer_other_kwargs["max_length"] - _TWO_BECAUSE_OF_SPECIAL_TOKEN
        elif truncation == NERTruncationScheme.SPLIT:
            assert max_length is not None
            assert stride is not None
            if add_special_tokens:
                assert 0 < stride <= (max_length - _TWO_BECAUSE_OF_SPECIAL_TOKEN), (stride, max_length)
            else:
                assert 0 < stride <= max_length, (stride, max_length)
            tokenizer_other_kwargs["truncation"] = False
            tokenizer_other_kwargs["max_length"] = None
        else:
            raise ValueError(truncation)

        enc = tokenizer(self.text, return_offsets_mapping=True, **tokenizer_other_kwargs)
        self.input_ids = enc["input_ids"]
        self.offset_mapping_start = [se[0] for se in enc["offset_mapping"]]
        self.offset_mapping_end = [se[1] for se in enc["offset_mapping"]]

        start_to_token_id = {t:i for i,t in enumerate(self.offset_mapping_start)}
        end_to_token_id = {t:i for i,t in enumerate(self.offset_mapping_end)}

        token_spans = list()
        offset_mapping_end_with_sentinel = [0] + list(self.offset_mapping_end)
        offset_mapping_start_with_sentinel = list(self.offset_mapping_start) + [self.offset_mapping_end[-1]]
        for span in self.spans:
            if fuzzy:
                for st in range(len(self.input_ids)):
                    if offset_mapping_end_with_sentinel[st] <= span.s < offset_mapping_end_with_sentinel[st+1]:
                        break
                else:
                    continue

                for et_minus_one in range(max(0,st-1), len(self.input_ids)):
                    if offset_mapping_start_with_sentinel[et_minus_one] < span.e <= offset_mapping_start_with_sentinel[et_minus_one+1]:
                        break
                else:
                    # continue
                    # raise ValueError(self)
                    pass

                token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))

            else:
                st = start_to_token_id.get(span.s, None)
                et_minus_one = end_to_token_id.get(span.e, None)
                if (st is not None) and (et_minus_one is not None):
                    token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))
        self.token_spans = token_spans

        if truncation == NERTruncationScheme.SPLIT:
            if add_special_tokens:
                outs = self._split_with_size(max_length=max_length-_TWO_BECAUSE_OF_SPECIAL_TOKEN, stride=stride)
                for out in outs:
                    out.with_special_tokens_(tokenizer=tokenizer)
            else:
                outs = self._split_with_size(max_length=max_length, stride=stride)
            return outs

        else:
            if add_special_tokens:
                self.with_special_tokens_(tokenizer=tokenizer)
            return self

    def with_special_tokens_(self, tokenizer:TokenizerInterface):
        assert not self.is_added_special_tokens, f'already special tokens are added. id:{self.id}'

        token_len_wo_sp_tokens = len(self.input_ids)
        new_input_ids = tokenizer.build_inputs_with_special_tokens(self.input_ids)
        assert len(new_input_ids) == token_len_wo_sp_tokens + _TWO_BECAUSE_OF_SPECIAL_TOKEN

        new_offset_mapping_start, new_offset_mapping_end, new_token_spans = self._padded_mappings_and_token_spans(num_forward_padding=_TWO_BECAUSE_OF_SPECIAL_TOKEN//2, num_backward_padding=_TWO_BECAUSE_OF_SPECIAL_TOKEN//2)

        self.input_ids = new_input_ids
        self.offset_mapping_start = new_offset_mapping_start
        self.offset_mapping_end = new_offset_mapping_end
        self.token_spans = new_token_spans
        self.is_added_special_tokens = True
        return self

    def with_special_tokens(self, tokenizer:TokenizerInterface):
        out = _copy.deepcopy(self)
        return out.with_special_tokens_(tokenizer=tokenizer)

    def with_query_and_special_tokens_(self, tokenizer:TokenizerInterface, encoded_query:_List[int], max_length:int, restrict_gold_class:_Optional[int]=None):
        assert not self.is_added_special_tokens, f'must be without special tokens. id:{self.id}'

        new_input_ids = tokenizer.build_inputs_with_special_tokens(encoded_query, self.input_ids)
        exceeded_len = max(0, len(new_input_ids) - max_length)
        if exceeded_len > 0:
            self._truncate_back_tokens_(size=exceeded_len)
            new_input_ids = tokenizer.build_inputs_with_special_tokens(encoded_query, self.input_ids)

        assert (self.input_ids[-1] == new_input_ids[-2]) and (self.input_ids[-1] != new_input_ids[-1]) # ... SOME_TOKEN [SEP]
        num_backward_padding = 1
        num_forward_padding = len(new_input_ids) - num_backward_padding - len(self.input_ids)
        new_offset_mapping_start, new_offset_mapping_end, new_token_spans = self._padded_mappings_and_token_spans(num_forward_padding=num_forward_padding, num_backward_padding=num_backward_padding)
        assert len(new_offset_mapping_start) == len(new_input_ids)

        self.input_ids = new_input_ids
        self.offset_mapping_start = new_offset_mapping_start
        self.offset_mapping_end = new_offset_mapping_end
        self.token_spans = new_token_spans
        self.is_added_special_tokens = True
        self.metadata["second_token_type_start"] = (_TWO_BECAUSE_OF_SPECIAL_TOKEN // 2) + len(encoded_query) + 1

        if restrict_gold_class is not None:
            self.spans = [span for span in self.spans if span.l == restrict_gold_class]
            self.token_spans = [token_span for token_span in self.token_spans if token_span.l == restrict_gold_class]

        return self

    def with_query_and_special_tokens(self, tokenizer:TokenizerInterface, encoded_query:_List[int], max_length:int, restrict_gold_class:_Optional[int]=None):
        out = _copy.deepcopy(self)
        return out.with_query_and_special_tokens_(tokenizer=tokenizer, encoded_query=encoded_query, max_length=max_length, restrict_gold_class=restrict_gold_class)

    def _split_with_size(self, max_length, stride):
        assert not self.is_added_special_tokens, f'must be without special tokens. id:{self.id}'

        num_splits = 1 + _math.ceil(max(0, len(self.input_ids) - max_length) / stride)

        if num_splits == 1:
            outs = [_copy.deepcopy(self)]
            outs[0].metadata["split"] = f's0/{num_splits}|c0|t0'
        else:
            outs = [_copy.deepcopy(self) for _ in range(num_splits)]
            for split_idx, out in enumerate(outs):
                token_start = stride * split_idx
                token_end = token_start + max_length
                out.input_ids = out.input_ids[token_start:token_end]
                out.offset_mapping_start = out.offset_mapping_start[token_start:token_end]
                out.offset_mapping_end = out.offset_mapping_end[token_start:token_end]

                char_start = out.offset_mapping_start[0]
                char_end = out.offset_mapping_end[-1]
                out.text = out.text[char_start:char_end]

                out.offset_mapping_start = [p - char_start for p in out.offset_mapping_start]
                out.offset_mapping_end = [p - char_start for p in out.offset_mapping_end]

                out.spans = [span for span in out.spans if (char_start<=span.s) and (span.e <= char_end)]
                out.spans = [_D.replace(span, s=span.s-char_start, e=span.e-char_start) for span in out.spans]

                out.token_spans = [token_span for token_span in out.token_spans if (token_start<=token_span.s) and (token_span.e <= token_end)]
                out.token_spans = [_D.replace(token_span, s=token_span.s-token_start, e=token_span.e-token_start) for token_span in out.token_spans]

                out.metadata["split"] = f's{split_idx}/{num_splits}|c{char_start}|t{token_start}'
        return outs

    def _padded_mappings_and_token_spans(self, num_forward_padding:int, num_backward_padding:int):
        last_position = self.offset_mapping_end[-1]
        padded_offset_mapping_start = [0 for _ in range(num_forward_padding)] + self.offset_mapping_start + [last_position for _ in range(num_backward_padding)]
        padded_offset_mapping_end = [0 for _ in range(num_forward_padding)] + self.offset_mapping_end + [last_position for _ in range(num_backward_padding)]

        padded_token_spans = list()
        for span in self.token_spans:
            new_s = span.s + num_forward_padding
            new_e = span.e + num_forward_padding
            copied_span = _D.replace(span, s=new_s, e=new_e)
            padded_token_spans.append(copied_span)

        return padded_offset_mapping_start, padded_offset_mapping_end, padded_token_spans

    def _truncate_back_tokens_(self, size:int):
        assert not self.is_added_special_tokens, 'cannot truncate after special tokens added. id:{self.id}'
        assert size >= 0
        self.input_ids = self.input_ids[:-size]
        self.offset_mapping_start = self.offset_mapping_start[:-size]
        self.offset_mapping_end = self.offset_mapping_end[:-size]

        new_token_len = len(self.input_ids)
        new_token_spans = list()
        for span in self.token_spans:
            # if span.e > new_token_len:
            #     continue
            if span.s >= new_token_len:
                continue
            if span.e > new_token_len:
                span = _D.replace(span, e=new_token_len)

            new_token_spans.append(span)

        self.token_spans = new_token_spans
        return self


    def get_sequence_label(self, tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, restrict_gold_class:_Optional[int]=None, num_class_without_negative=None, strict:bool=True) -> _Union[_List[int],_List[_List[int]]]:
        """
        output := [label_0, label_1, label_2, ...]

        if label_scheme==SINGLE_LABEL:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:L-class_0, 4:U-class_0, 5:B-class_1, 6:I-class_1, ..., 4*n+1:B-class_n, 4*n+2:I-class_n, 4*n+3:L-class_n, 4*n+4:U-class_n, ...}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:B-class_1, 4:I-class_1, ..., 2*n+1:B-class_n, 2*n+2:I-class_n, ...}
            if tagging_scheme==TOKEN_LEVEL:
                label_t \in {0:O, 1:class_0, 2:class_1, ..., n+1:class_n, ...}

        if label_scheme==MULTI_LABEL:
            label_t := [label_t_class_0, label_t_class_1, ..., label_t_class_N]
            N: num_class_without_negative

            label_t_class_k := int
            k: the index of the class

            if tagging_scheme==BILOU:
                label_t_class_k \in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t_class_k \in {0:O, 1:B, 2:I}
            if tagging_scheme==TOKEN_LEVEL:
                label_t_class_k \in {0:Negative, 1:Positive}

        if label_scheme==SPAN_ONLY:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B, 2:I}
            if tagging_scheme==TOKEN_LEVEL:
                label_t \in {0:Negative, 1:Positive}
        """
        if label_scheme == NERLabelScheme.MULTI_LABEL:
            assert num_class_without_negative is not None, "num_class_without_negative must be specified under the multi-labelling setting."
        if restrict_gold_class is None:
            target_spans = self.token_spans
        else:
            target_spans = [span for span in self.token_spans if span.l == restrict_gold_class]

        if label_scheme in [NERLabelScheme.SINGLE_LABEL, NERLabelScheme.SPAN_ONLY]:
            out = [int(NERSpanTag.O) for _ in range(len(self.input_ids))]
        elif label_scheme == NERLabelScheme.MULTI_LABEL:
            transposed_out = [[int(NERSpanTag.O) for _ in range(len(self.input_ids))] for _ in range(num_class_without_negative)]
        else:
            raise ValueError(label_scheme)

        for span in target_spans:
            if tagging_scheme == NERTaggingScheme.BILOU:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), f'there must not be overlapped spans: {target_spans}'
                    target_out = out
                    class_offset = span.l * 4 # 4 <- len({B, I, L, U})
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), f'there must not be overlapped spans: {target_spans}'
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[span.l]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                if span.s + 1 == span.e:
                    target_out[span.s] = NERSpanTag.U + class_offset
                else:
                    target_out[span.s] = NERSpanTag.B + class_offset
                    target_out[span.e-1] = NERSpanTag.L + class_offset
                    for i in range(span.s+1, span.e-1):
                        target_out[i] = NERSpanTag.I + class_offset

            elif tagging_scheme == NERTaggingScheme.BIO:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), f'there must not be overlapped spans: {target_spans}'
                    target_out = out
                    class_offset = span.l * 2 # 2 <- len({B, I})
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), f'there must not be overlapped spans: {target_spans}'
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[span.l]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                target_out[span.s] = NERSpanTag.B + class_offset
                for i in range(span.s+1, span.e):
                    target_out[i] = NERSpanTag.I + class_offset

            elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
                if label_scheme == NERLabelScheme.SINGLE_LABEL:
                    target_out = out
                    class_offset = span.l
                elif label_scheme == NERLabelScheme.SPAN_ONLY:
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MULTI_LABEL:
                    target_out = transposed_out[span.l]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                for i in range(span.s, span.e):
                    target_out[i] = 1 + class_offset

            else:
                raise ValueError(tagging_scheme)

        if label_scheme == NERLabelScheme.MULTI_LABEL:
            out = [list(v) for v in zip(*transposed_out)]

        return out

    def decode_token_span_to_char_span(self, span:_Union[NERSpan, _List[NERSpan]], strip:bool=False, recover_split:bool=False) -> _Union[NERSpan, _List[NERSpan]]:
        if not isinstance(span, NERSpan):
            return [self.decode_token_span_to_char_span(s, strip=strip, recover_split=recover_split) for s in span]

        char_start = self.offset_mapping_start[span.s]
        if span.s == span.e:
            char_end = char_start
        else:
            char_end = self.offset_mapping_end[span.e-1]
        out = NERSpan(s=char_start, e=char_end, l=span.l, id=span.id)

        if strip:
            out = self.strip_char_spans(out)
        if recover_split:
            out = self.recover_split_offset_of_char_spans(out)
        return out

    def strip_char_spans(self, span:_Union[NERSpan, _List[NERSpan]]) -> _Union[NERSpan, _List[NERSpan]]:
        if not isinstance(span, NERSpan):
            return [self.strip_char_spans(s) for s in span]

        forward_blank_span = _re.match("^\\s*", self.text[span.s:span.e]).span()
        forward_blank_size = forward_blank_span[1] - forward_blank_span[0]
        backward_blank_span = _re.search("\\s*$", self.text[span.s+forward_blank_size:span.e]).span()
        backward_blank_size = backward_blank_span[1] - backward_blank_span[0]
        out = _D.replace(span, s=span.s+forward_blank_size, e=span.e-backward_blank_size)
        return out

    def recover_split_offset_of_char_spans(self, span:_Union[NERSpan, _List[NERSpan]]) -> _Union[NERSpan, _List[NERSpan]]:
        if not isinstance(span, NERSpan):
            return [self.recover_split_offset_of_char_spans(s) for s in span]

        split_offset = self.get_decoded_metadata()["split"]["char_start"]
        out = _D.replace(span, s=span.s+split_offset, e=span.e+split_offset)
        return out

    def get_decoded_metadata(self):
        outs = dict(self.metadata)
        if "split" in outs:
            data = {col[0]:col[1:] for col in outs["split"].split("|")}
            split_idx, num_splits = map(int, data["s"].split("/"))
            outs["split"] = {
                "num_splits": num_splits,
                "split_idx": split_idx,
                "char_start": int(data["c"]),
                "token_start": int(data["t"]),
            }
        return outs



# %%
def convert_sequence_label_to_spans(sequence_label:_Union[_List[int],_List[_List[int]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL) -> _List[NERSpan]:
    if label_scheme in [NERLabelScheme.SINGLE_LABEL, NERLabelScheme.SPAN_ONLY]:
        if tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
            return [NERSpan(s=t,e=t+1,l=label-1) for t, label in enumerate(sequence_label) if label != 0]

        elif tagging_scheme == NERTaggingScheme.BILOU:
            out = list()
            start = None
            class_ = None

            for t, label in enumerate(sequence_label):
                if label == 0: # "O"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t}. treat as termination.: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        start = None
                        class_ = None

                elif (label-1) % 4 == 0: # "B"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t}. treat as termination.: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        # start = None
                        # class_ = None
                    start = t
                    class_ = (label-1) // 4

                elif (label-1) % 4 == 1: # "I"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 4

                    # assert class_ == ((label-1) // 4)
                    if class_ != ((label-1) // 4):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 4

                elif (label-1) % 4 == 2: # "L"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 4

                    # assert class_ == ((label-1) // 4)
                    if class_ != ((label-1) // 4):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 4

                    out.append(NERSpan(s=start, e=t+1, l=class_))
                    start = None
                    class_ = None

                elif (label-1) % 4 == 3: # "U"
                    # assert start is None
                    if start is not None:
                        _logger.warning(f'span ends without "L" at timestep {t}. treat as termination.: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        start = None
                        class_ = None
                    out.append(NERSpan(s=t, e=t+1, l=(label-1)//4))

                else:
                    raise ValueError(label)

            # assert start is None
            if start is not None:
                _logger.warning(f'span ends without "L" at timestep {t+1}. treat as termination.: {sequence_label}')
                out.append(NERSpan(s=start, e=t+1, l=class_))
                start = None
                class_ = None

            return out

        elif tagging_scheme == NERTaggingScheme.BIO:
            out = list()
            start = None
            class_ = None

            for t, label in enumerate(sequence_label):
                if label == 0: # "O"
                    if start is not None:
                        out.append(NERSpan(s=start, e=t, l=class_))
                        start = None
                        class_ = None

                elif (label-1) % 2 == 0: # "B"
                    if start is not None:
                        out.append(NERSpan(s=start, e=t, l=class_))
                        # start = None
                        # class_ = None
                    start = t
                    class_ = (label-1) // 2

                elif (label-1) % 2 == 1: # "I"
                    # assert start is not None
                    if start is None:
                        _logger.warning(f'span starts without "B" at timestep {t}. treat as new beginning.: {sequence_label}')
                        start = t
                        class_ = (label-1) // 2

                    # assert class_ == ((label-1) // 2)
                    if class_ != ((label-1) // 2):
                        _logger.warning(f'span class is incosistent at timestep {t}. treat as new beginning: {sequence_label}')
                        out.append(NERSpan(s=start, e=t, l=class_))
                        # start = None
                        # class_ = None
                        start = t
                        class_ = (label-1) // 2

                else:
                    raise ValueError(label)

            if start is not None:
                out.append(NERSpan(s=start, e=t+1, l=class_))
                start = None
                class_ = None

            return out

        else:
            raise ValueError(tagging_scheme)

    elif label_scheme == NERLabelScheme.MULTI_LABEL:
        num_class = len(sequence_label[0])
        outs = list()
        for c in range(num_class):
            sequence_label_class_c = [labels[c] for labels in sequence_label]
            spans_class_c = convert_sequence_label_to_spans(sequence_label=sequence_label_class_c, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SPAN_ONLY)
            outs.extend([NERSpan(s=span.s, e=span.e, l=c) for span in spans_class_c])
        return outs

    else:
        raise ValueError(label_scheme)

def viterbi_decode(logits_sequence:_Union[_List[float],_List[_List[float]],_List[_List[_List[float]]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, scalar_logit_for_token_level:bool=False, as_spans:bool=False) -> _Union[_Union[_List[int],_List[_List[int]]], _List[NERSpan]]:
    """
    input: logits_sequence
    - 2D or 3D float list. shape==[seq_len, [num_class,] num_label].

    output: sequence_label

    if (scalar_logit_for_token_level == True) and (tagging_scheme == TOKEN_LEVEL) and (label_scheme in [MULTI_LABEL, SPAN_ONLY]),
    the expected shape of logits_sequence is [seq_len, [num_class,]] and each value logits_sequence[i[,j]] will be treated as a logit of the positive probability.
    otherwise, the shape should be [seq_len, [num_class,] 2] (2:(logit_negative, logit_positive)).


    if label_scheme==SINGLE_LABEL:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_L-class_0, logit_U-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*4 + 1 (4 <= {B,I,L,U})

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*2 + 1 (2 <= {B,I})

        if tagging_scheme==TOKEN_LEVEL
            logits_time-step_t := [logit_O, logit_class_0, logit_class_1, ...]
            len(logits_time-step_t) == num_class*1 + 1 (1 <= {Positive})


    if label_scheme==MULTI_LABEL:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]
        logits_time-step_t := [logits_class_0, logits_class_1, ...]

        if tagging_scheme==BILOU:
            logits_class_c := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_class_c := [logit_O, logit_B, logit_I]

        if tagging_scheme==TOKEN_LEVEL
            logits_class_c := [logit_Negative, logit_Positive]


    if label_scheme==SPAN_ONLY:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B, logit_I]

        if tagging_scheme==TOKEN_LEVEL
            logits_time-step_t := [logit_Negative, logit_Positive]
    """
    if as_spans:
        sequence_label = viterbi_decode(logits_sequence=logits_sequence, tagging_scheme=tagging_scheme, label_scheme=label_scheme, scalar_logit_for_token_level=scalar_logit_for_token_level, as_spans=False)
        return convert_sequence_label_to_spans(sequence_label=sequence_label, tagging_scheme=tagging_scheme, label_scheme=label_scheme)


    logits_sequence:_numpy.ndarray = _numpy.array(logits_sequence, dtype=_numpy.float32)

    if label_scheme == NERLabelScheme.SINGLE_LABEL:
        assert len(logits_sequence.shape) == 2, logits_sequence.shape

        if tagging_scheme == NERTaggingScheme.BILOU:
            num_class = (logits_sequence.shape[1] - 1) // 4
        elif tagging_scheme == NERTaggingScheme.BIO:
            num_class = (logits_sequence.shape[1] - 1) // 2
        elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
            num_class = logits_sequence.shape[1] - 1
        else:
            raise ValueError(tagging_scheme)
    elif label_scheme == NERLabelScheme.MULTI_LABEL:
        if (tagging_scheme == NERTaggingScheme.TOKEN_LEVEL) and scalar_logit_for_token_level:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 3, logits_sequence.shape

        decoded = list()
        for class_i in range(logits_sequence.shape[1]):
            logit_sequence_class_i = logits_sequence[:,class_i]
            decoded_class_i = viterbi_decode(logits_sequence=logit_sequence_class_i, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SPAN_ONLY, scalar_logit_for_token_level=scalar_logit_for_token_level)
            decoded.append(decoded_class_i)
        decoded = list(zip(*decoded)) # [num_class, seq_len] -> [seq_len, num_class]
        return decoded
    elif label_scheme == NERLabelScheme.SPAN_ONLY:
        if (tagging_scheme == NERTaggingScheme.TOKEN_LEVEL) and scalar_logit_for_token_level:
            assert len(logits_sequence.shape) == 1, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape

        num_class = 1
    else:
        raise ValueError(label_scheme)

    if tagging_scheme == NERTaggingScheme.BILOU:
        num_label_wo_O = 4

        transition_paths = [None for _ in range(1+num_label_wo_O*num_class)]
        terminations = [0] + [num_label_wo_O*c + L_or_U for c in range(num_class) for L_or_U in [3,4]]
        transition_paths[0] = "from-termination"
        for c in range(num_class):
            # B
            transition_paths[num_label_wo_O*c+1] = "from-termination"
            # I
            transition_paths[num_label_wo_O*c+2] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I
            # L
            transition_paths[num_label_wo_O*c+3] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I
            # U
            transition_paths[num_label_wo_O*c+4] = "from-termination"

    elif tagging_scheme == NERTaggingScheme.BIO:
        num_label_wo_O = 2

        transition_paths = [None for _ in range(1+num_label_wo_O*num_class)]
        terminations = list(range(1+num_label_wo_O*num_class))
        transition_paths[0] = "from-termination"
        for c in range(num_class):
            # B
            transition_paths[num_label_wo_O*c+1] = "from-termination"
            # I
            transition_paths[num_label_wo_O*c+2] = [num_label_wo_O*c+1, num_label_wo_O*c+2] # from B or I

    elif tagging_scheme == NERTaggingScheme.TOKEN_LEVEL:
        if label_scheme == NERLabelScheme.SPAN_ONLY and scalar_logit_for_token_level:
            return (logits_sequence > 0).astype(_numpy.int32).tolist()
        else:
            return logits_sequence.argmax(-1).tolist()

    else:
        raise ValueError(tagging_scheme)

    # init
    past_trajs_history = list()
    cumsum_logits_history = list()
    cumsum_logits_history.append([0.0] + [-_numpy.inf] * (num_label_wo_O*num_class)) # sentinel

    # forward
    for timestep in range(logits_sequence.shape[0]):
        logits = logits_sequence[timestep]

        prev_cumsum_logits = cumsum_logits_history[-1]
        cumsum_logits = [None for _ in range(1+num_label_wo_O*num_class)]
        past_trajs = [None for _ in range(1+num_label_wo_O*num_class)]

        prev_best_terminal_index = max(terminations, key=lambda i:prev_cumsum_logits[i])
        prev_best_terminal_cumsum_logit = prev_cumsum_logits[prev_best_terminal_index]

        for l in range(1+num_label_wo_O*num_class):
            if transition_paths[l] == "from-termination": # O/B/U
                past_trajs[l] = prev_best_terminal_index
                cumsum_logits[l] = logits[l] + prev_best_terminal_cumsum_logit
            else: # I/L
                past_traj = max(transition_paths[l], key=lambda i:prev_cumsum_logits[i])
                past_trajs[l] = past_traj
                cumsum_logits[l] = logits[l] + prev_cumsum_logits[past_traj]

        cumsum_logits_history.append(cumsum_logits)
        past_trajs_history.append(past_trajs)

    # backward
    last_cumsum_logits = cumsum_logits_history[-1]
    traj = max(terminations, key=lambda i:last_cumsum_logits[i])
    reverse_trajectory = list()
    for past_trajs in reversed(past_trajs_history):
        reverse_trajectory.append(traj)
        traj = past_trajs[traj]
    sequence_label = list(reversed(reverse_trajectory))
    return sequence_label

def merge_spans(spans:_List[NERSpan]) -> _List[NERSpan]:
    if len(spans) == 0:
        return list()
    max_end = max([span.e for span in spans])
    zero_table = [0 for _ in range(max_end+1)]
    label_to_table = _collections.defaultdict(lambda: list(zero_table))
    for span in spans:
        for i in range(span.s, span.e):
            label_to_table[span.l][i] = 1

    outs = list()
    for label, table in label_to_table.items():
        t = 0
        while t < max_end:
            if table[t] == 1:
                start = t
                while (t < max_end) and (table[t] == 1):
                    t += 1
                end = t
                outs.append(NERSpan(s=start,e=end,l=label))
                continue
            else:
                t += 1
                continue

    return outs


# %%
if __name__ == "__main__":
    tok = _transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    # %%
    _instance_without_sp = NERInstance.build(
        text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
        spans = [[9,9+14, 0, "first-span:class_0:very long text"], [88,88+10, 3, "second-span:class_3:the bert model"], (116,116+9, 2, "third-span:class_2:the input")],
    )
    _num_class = 4
    _instance_without_sp.encode_(tokenizer=tok, add_special_tokens=False, truncation=False)
    _instance_with_sp = _instance_without_sp.with_special_tokens(tok)
    _bilou_gold_label = _instance_with_sp.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL)
    _multi_label_indep_gold_label = _instance_with_sp.get_sequence_label(tagging_scheme=NERTaggingScheme.TOKEN_LEVEL, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=_num_class)
    print("_instance_without_sp:", _instance_with_sp)
    print("_bilou_gold_label:", _bilou_gold_label)
    print("_multi_label_indep_gold_label:", _multi_label_indep_gold_label)


    # %%
    _encoded_query = tok("what model is used ?", add_special_tokens=False)["input_ids"]
    _instance_with_query = _instance_without_sp.with_query_and_special_tokens(tokenizer=tok, encoded_query=_encoded_query, max_length=30, restrict_gold_class=3)
    _gold_label_for_query = _instance_with_query.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY)
    print(_instance_with_query)
    print()
    print("label:", _gold_label_for_query)


    # %%
    _instances_split_by_stride = NERInstance.build(
        text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
        spans = [[9,9+14, 0, "first-span:class_0:very long text"], [88,88+10, 3, "second-span:class_3:the bert model"], (116,116+9, 2, "third-span:class_2:the input")],
        tokenizer = tok,
        add_special_tokens=True,
        truncation = NERTruncationScheme.SPLIT,
        max_length = 8,
        stride = 2,
    )
    print(type(_instances_split_by_stride), type(_instances_split_by_stride[0]), len(_instances_split_by_stride))

    # %%
    # step_preds = BNER.viterbi_decode(step_logit, tagging_scheme=model_config.tagging_scheme, label_scheme=model_config.label_scheme, scalar_logit_for_independent=True, as_spans=True)
    # step_preds = instance.decode_token_span_to_char_span(step_preds, strip=True, recover_split=True)


# %%
