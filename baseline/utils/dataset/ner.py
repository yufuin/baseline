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
@_D.dataclass
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

    def __eq__(lhs, rhs) -> bool:
        return (lhs.s, lhs.e, lhs.l) == (rhs.s, rhs.e, rhs.l)

NERSpanAsList = _Union[_Tuple[int,int,int], _Tuple[int,int,int,_Any]]

class NERTaggingScheme(str, _enum.Enum):
    BILOU = "BILOU"
    BIO = "BIO"
    INDEPENDENT = "INDEPENDENT"

class NERLabelScheme(str, _enum.Enum):
    SINGLE_LABEL = "SINGLE_LABEL"
    MULTI_LABEL = "MULTI_LABEL"
    SPAN_ONLY = "SPAN_ONLY"

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
            out.encode_(tokenizer, **encode_func_args)
        if check_some:
            if type(out) in [list, tuple]:
                [each_out.check_some() for each_out in out]
            else:
                out.check_some()
        return out

    def check_some(self):
        for span in self.spans:
            assert span.s < span.e, span
        if self.token_spans is not None:
            for span in self.token_spans:
                assert span.s < span.e, span
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
                    raise ValueError(self)

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

    def with_query_and_special_tokens_(self, tokenizer:TokenizerInterface, encoded_query:_List[int], max_length:int):
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
        return self

    def with_query_and_special_tokens(self, tokenizer:TokenizerInterface, encoded_query:_List[int], max_length:int):
        out = _copy.deepcopy(self)
        return out.with_query_and_special_tokens_(tokenizer=tokenizer, encoded_query=encoded_query, max_length=max_length)

    def _split_with_size(self, max_length, stride):
        assert not self.is_added_special_tokens, f'must be without special tokens. id:{self.id}'

        num_splits = 1 + _math.ceil(max(0, len(self.input_ids) - max_length) / stride)

        if num_splits == 1:
            outs = [_copy.deepcopy(self)]
            outs[0].metadata["split"] = f'c0|s0/{num_splits}|t0'
        else:
            outs = [_copy.deepcopy(self) for _ in range(num_splits)]
            for s, out in enumerate(outs):
                token_start = stride * s
                token_end = token_start + max_length
                out.input_ids = out.input_ids[token_start:token_end]
                out.offset_mapping_start = out.offset_mapping_start[token_start:token_end]
                out.offset_mapping_end = out.offset_mapping_end[token_start:token_end]
                out.token_spans = [token_span for token_span in out.token_spans if (token_start<=token_span.s) and (token_span.e <= token_end)]

                char_start = out.offset_mapping_start[0]
                char_end = out.offset_mapping_end[-1]
                out.text = out.text[char_start:char_end]
                out.spans = [span for span in out.spans if (char_start<=span.s) and (span.e <= char_end)]

                out.offset_mapping_start = [p - char_start for p in out.offset_mapping_start]
                out.offset_mapping_end = [p - char_start for p in out.offset_mapping_end]
                for span in out.spans:
                    span.s = span.s - char_start
                    span.e = span.e - char_start
                for token_span in out.token_spans:
                    token_span.s = token_span.s - token_start
                    token_span.e = token_span.e - token_start

                out.metadata["split"] = f'c{char_start}|s{s}/{num_splits}|t{token_start}'
        return outs

    def _padded_mappings_and_token_spans(self, num_forward_padding:int, num_backward_padding:int):
        last_position = self.offset_mapping_end[-1]
        padded_offset_mapping_start = [0 for _ in range(num_forward_padding)] + self.offset_mapping_start + [last_position for _ in range(num_backward_padding)]
        padded_offset_mapping_end = [0 for _ in range(num_forward_padding)] + self.offset_mapping_end + [last_position for _ in range(num_backward_padding)]

        padded_token_spans = list()
        for span in self.token_spans:
            copied_span = _copy.deepcopy(span)
            copied_span.s += num_forward_padding
            copied_span.e += num_forward_padding
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
            copied_span = NERSpan(**_D.asdict(span))
            if copied_span.s >= new_token_len:
                continue
            if copied_span.e > new_token_len:
                copied_span.e = new_token_len
            new_token_spans.append(copied_span)
        self.token_spans = new_token_spans
        return self


    def get_sequence_label(self, tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, only_label:_Optional[int]=None, num_class_without_negative=None, strict:bool=True) -> _Union[_List[int],_List[_List[int]]]:
        """
        output := [label_0, label_1, label_2, ...]

        if label_scheme==SINGLE_LABEL:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:L-class_0, 4:U-class_0, 5:B-class_1, 6:I-class_1, ..., 4*n+1:B-class_n, 4*n+2:I-class_n, 4*n+3:L-class_n, 4*n+4:U-class_n, ...}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:B-class_1, 4:I-class_1, ..., 2*n+1:B-class_n, 2*n+2:I-class_n, ...}
            if tagging_scheme==INDEPENDENT:
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
            if tagging_scheme==INDEPENDENT:
                label_t_class_k \in {0:Negative, 1:Positive}

        if label_scheme==SPAN_ONLY:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B, 2:I}
            if tagging_scheme==INDEPENDENT:
                label_t \in {0:Negative, 1:Positive}
        """
        if label_scheme == NERLabelScheme.MULTI_LABEL:
            assert num_class_without_negative is not None, "num_class_without_negative must be specified under the multi-labelling setting."
        if only_label is None:
            target_spans = self.token_spans
        else:
            target_spans = [span for span in self.token_spans if span.l == only_label]

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

            elif tagging_scheme == NERTaggingScheme.INDEPENDENT:
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

        out = _copy.deepcopy(span)
        while (out.s < out.e) and _re.match("\\s", self.text[out.s]):
            out.s += 1
        while (out.s < out.e) and _re.match("\\s", self.text[out.e-1]):
            out.e -= 1
        return out

    def recover_split_offset_of_char_spans(self, span:_Union[NERSpan, _List[NERSpan]]) -> _Union[NERSpan, _List[NERSpan]]:
        if not isinstance(span, NERSpan):
            return [self.recover_split_offset_of_char_spans(s) for s in span]

        split_offset = [int(data[1:]) for data in self.metadata["split"].split("|") if data[0] == "c"]
        assert len(split_offset) == 1
        split_offset = split_offset[0]

        out = _copy.deepcopy(span)
        out.s = out.s + split_offset
        out.e = out.e + split_offset
        return out

    def get_parsed_metadata(self):
        outs = dict(self.metadata)
        if "split" in outs:
            data = {col[0]:col[1:] for col in outs["split"].split("|")}
            s, num_splits = map(int, data["s"].split("/"))
            outs["split"] = {
                "char_start": int(data["c"]),
                "token_start": int(data["t"]),
                "num_splits": num_splits,
                "s": s,
            }
        return outs



# %%
def convert_sequence_label_to_spans(sequence_label:_Union[_List[int],_List[_List[int]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL) -> _List[NERSpan]:
    if label_scheme in [NERLabelScheme.SINGLE_LABEL, NERLabelScheme.SPAN_ONLY]:
        if tagging_scheme == NERTaggingScheme.INDEPENDENT:
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

def viterbi_decode(logits_sequence:_Union[_List[float],_List[_List[float]],_List[_List[_List[float]]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SINGLE_LABEL, scalar_logit_for_independent:bool=False, as_spans:bool=False) -> _Union[_Union[_List[int],_List[_List[int]]], _List[NERSpan]]:
    """
    input: logits_sequence
    - 2D or 3D float list. shape==[seq_len, [num_class,] num_label].

    output: sequence_label

    if (scalar_logit_for_independent == True) and (tagging_scheme == INDEPENDENT) and (label_scheme in [MULTI_LABEL, SPAN_ONLY]),
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

        if tagging_scheme==INDEPENDENT
            logits_time-step_t := [logit_O, logit_class_0, logit_class_1, ...]
            len(logits_time-step_t) == num_class*1 + 1 (1 <= {Positive})


    if label_scheme==MULTI_LABEL:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]
        logits_time-step_t := [logits_class_0, logits_class_1, ...]

        if tagging_scheme==BILOU:
            logits_class_c := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_class_c := [logit_O, logit_B, logit_I]

        if tagging_scheme==INDEPENDENT
            logits_class_c := [logit_Negative, logit_Positive]


    if label_scheme==SPAN_ONLY:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B, logit_I]

        if tagging_scheme==INDEPENDENT
            logits_time-step_t := [logit_Negative, logit_Positive]
    """
    if as_spans:
        sequence_label = viterbi_decode(logits_sequence=logits_sequence, tagging_scheme=tagging_scheme, label_scheme=label_scheme, scalar_logit_for_independent=scalar_logit_for_independent, as_spans=False)
        return convert_sequence_label_to_spans(sequence_label=sequence_label, tagging_scheme=tagging_scheme, label_scheme=label_scheme)


    logits_sequence:_numpy.ndarray = _numpy.array(logits_sequence, dtype=_numpy.float32)

    if label_scheme == NERLabelScheme.SINGLE_LABEL:
        assert len(logits_sequence.shape) == 2, logits_sequence.shape

        if tagging_scheme == NERTaggingScheme.BILOU:
            num_class = (logits_sequence.shape[1] - 1) // 4
        elif tagging_scheme == NERTaggingScheme.BIO:
            num_class = (logits_sequence.shape[1] - 1) // 2
        elif tagging_scheme == NERTaggingScheme.INDEPENDENT:
            num_class = logits_sequence.shape[1] - 1
        else:
            raise ValueError(tagging_scheme)
    elif label_scheme == NERLabelScheme.MULTI_LABEL:
        if (tagging_scheme == NERTaggingScheme.INDEPENDENT) and scalar_logit_for_independent:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 3, logits_sequence.shape

        decoded = list()
        for class_i in range(logits_sequence.shape[1]):
            logit_sequence_class_i = logits_sequence[:,class_i]
            decoded_class_i = viterbi_decode(logits_sequence=logit_sequence_class_i, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SPAN_ONLY, scalar_logit_for_independent=scalar_logit_for_independent)
            decoded.append(decoded_class_i)
        decoded = list(zip(*decoded)) # [num_class, seq_len] -> [seq_len, num_class]
        return decoded
    elif label_scheme == NERLabelScheme.SPAN_ONLY:
        if (tagging_scheme == NERTaggingScheme.INDEPENDENT) and scalar_logit_for_independent:
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

    elif tagging_scheme == NERTaggingScheme.INDEPENDENT:
        if label_scheme == NERLabelScheme.SPAN_ONLY and scalar_logit_for_independent:
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
    instance = NERInstance.build(
        text = "This is biomedical example.",
        spans = [[0,4, 0, "first-span:class_0"], [5,7, 1, "second-span:class_1"], (8,27, 0, "third-span:class_0")],
        tokenizer = tok,
        add_special_tokens=False,
    )
    instance.with_special_tokens_(tok)
    print("instance:", instance)
    print()

    # %%
    print("multi class single labelling sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SINGLE_LABEL))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SINGLE_LABEL))
    print()

    # %%
    print("multi labelling sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2))
    print()

    # %%
    print("span-only (no-class) sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY))
    print("BILOU only for class_0 ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY, only_label=0))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SPAN_ONLY))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SPAN_ONLY))
    print()

    # %%
    print(instance.token_spans)
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SINGLE_LABEL))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SINGLE_LABEL), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SINGLE_LABEL))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SINGLE_LABEL), tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SINGLE_LABEL))

    # %%
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MULTI_LABEL))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MULTI_LABEL))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.MULTI_LABEL, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.MULTI_LABEL))

    # %%
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SPAN_ONLY))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SPAN_ONLY), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SPAN_ONLY))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SPAN_ONLY), tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.SPAN_ONLY))

    # %%
    positive_logits = _numpy.array([
        [-0.12580859661102295, 0.13866497576236725, -0.004946088418364525, -0.011039067059755325, 0.04118518531322479, 0.209847092628479, -0.240921288728714, -0.026520986109972, -0.07723920792341232, 0.006692873779684305, -0.3117348253726959],
        [1.0395761728286743, -1.139390230178833, -1.67435884475708, -0.682197630405426, -0.005922921001911163, 0.4142416715621948, -0.8619325160980225, -0.527134120464325, 0.5418972969055176, 0.4669799208641052, 0.9111515283584595],
        [0.9126020669937134, -0.8182739019393921, -1.1124005317687988, -0.31609854102134705, -0.5066011548042297, -0.5548560619354248, -0.5321623682975769, -0.0609733872115612, 1.052380084991455, 0.69451904296875, 0.6331266760826111],
        [1.1847517490386963, -0.2960182726383209, 0.17322547733783722, -0.9496498107910156, -0.7404254078865051, -0.41485345363616943, 0.5226980447769165, 0.3296244442462921, 0.5696917772293091, 0.14669597148895264, -0.407884806394577],
        [-0.048488348722457886, -0.028185393661260605, -0.45448189973831177, -0.571286141872406, 0.3106667101383209, -0.649383544921875, 0.14023838937282562, 0.0034116366878151894, 0.7123101353645325, -0.35180336236953735, -0.7513846755027771],
        [1.1311267614364624, -1.0243971347808838, -0.9930524230003357, -0.3362758159637451, -0.5516917109489441, 0.08782649040222168, -0.3966140151023865, 0.19338271021842957, 0.9406710863113403, -0.07828716933727264, 0.5972633361816406],
        [0.9672456979751587, -1.850893497467041, -1.0942933559417725, -0.9000891447067261, -0.6152650713920593, 0.15397143363952637, -0.4109809100627899, -0.003905489109456539, 2.078101634979248, -0.46680164337158203, 0.4010751247406006]
    ]) # [seq_len=7, num_class=11]
    logits = _numpy.stack([-positive_logits, positive_logits], axis=-1) # [7, 11, 2(negative/positive)]
    probs = 1 / (1+_numpy.exp(-logits))

    decoded1 = viterbi_decode(logits, tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.MULTI_LABEL)
    decoded2 = viterbi_decode(positive_logits, tagging_scheme=NERTaggingScheme.INDEPENDENT, label_scheme=NERLabelScheme.MULTI_LABEL, scalar_logit_for_independent=True)
    decoded1 == decoded2, _numpy.array(decoded1).shape, decoded1

    # %%
    independent_spans = [
        NERSpan(s=3, e=4, l=99, id=None),
        NERSpan(s=85, e=86, l=100, id=None),
        NERSpan(s=86, e=87, l=100, id=None),
        NERSpan(s=87, e=88, l=100, id=None),
        NERSpan(s=88, e=89, l=100, id=None),
        NERSpan(s=89, e=90, l=100, id=None),
        NERSpan(s=10, e=11, l=101, id=None),
        NERSpan(s=11, e=12, l=101, id=None),
        NERSpan(s=12, e=13, l=101, id=None),
        NERSpan(s=13, e=14, l=101, id=None),
        NERSpan(s=14, e=15, l=101, id=None),
        NERSpan(s=18, e=19, l=101, id=None),
        NERSpan(s=19, e=20, l=101, id=None),
    ]
    merged_spans = merge_spans(independent_spans)
    merged_spans

# %%
