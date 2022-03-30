# %%
import logging as _logging
_logger = _logging.getLogger(__name__)
_logger.setLevel(_logging.WARNING)
_ch = _logging.StreamHandler()
_ch.setLevel(_logging.WARNING)
_formatter = _logging.Formatter('%(name)s - %(levelname)s:%(message)s')
_ch.setFormatter(_formatter)
_logger.addHandler(_ch)

__TWO_BECAUSE_OF_SPECIAL_TOKEN = 2


import collections as _collections
import dataclasses as _D
import enum as _enum
from typing import List as _List, Optional as _Optional, Tuple as _Tuple, Union as _Union, Any as _Any

import numpy as _numpy
import transformers as _transformers

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
NERSpanAsList = _Union[_Tuple[int,int,int], _Tuple[int,int,int,_Any]]

class NERTaggingScheme(int, _enum.Enum):
    BILOU = 0
    BIO = _enum.auto()
    Independent = _enum.auto()

class NERLabelScheme(int, _enum.Enum):
    SingleLabel = 0
    MultiLabel = _enum.auto()
    SpanOnly = _enum.auto()

class NERSpanTag(int, _enum.Enum):
    O = 0
    B = _enum.auto()
    I = _enum.auto()
    L = _enum.auto()
    U = _enum.auto()

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

    meta_data: _Any = None

    @classmethod
    def load_from_dict(cls, dumped):
        dumped = dict(dumped)
        dumped["spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["spans"]]
        if isinstance(dumped["token_spans"], list):
            dumped["token_spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["token_spans"]]
        return cls(**dumped)

    @classmethod
    def build(cls, text:str, spans:_List[_Union[NERSpan,NERSpanAsList]], id:_Any=None, *, check_some:bool=True, tokenizer:_Optional[_transformers.PreTrainedTokenizer]=None, add_special_tokens:_Optional[bool]=None, fuzzy:_Optional[bool]=None, tokenizer_other_kwargs:_Optional[dict]=None):
        spans = [span if type(span) is NERSpan else NERSpan(*span) for span in spans]
        out = cls(text=text, spans=spans, id=id)
        if tokenizer is not None:
            encode_func_args = dict()
            if add_special_tokens is not None:
                encode_func_args["add_special_tokens"] = add_special_tokens
            if fuzzy is not None:
                encode_func_args["fuzzy"] = fuzzy
            if tokenizer_other_kwargs is not None:
                encode_func_args["tokenizer_other_kwargs"] = tokenizer_other_kwargs
            out.encode_(tokenizer, **encode_func_args)
        if check_some:
            out.check_some()
        return out

    def check_some(self):
        for span in self.spans:
            assert span.s < span.e, span
        if self.token_spans is not None:
            for span in self.token_spans:
                assert span.s < span.e, span
        return self

    def encode_(self, tokenizer:_transformers.PreTrainedTokenizer, *, add_special_tokens:bool=False, fuzzy:bool=True, tokenizer_other_kwargs:_Optional[dict]=None):
        # reset
        self.is_added_special_tokens = False

        if tokenizer_other_kwargs is None:
            tokenizer_other_kwargs = dict()
        else:
            tokenizer_other_kwargs = dict(tokenizer_other_kwargs)
        if "add_special_tokens" in tokenizer_other_kwargs:
            _logger.warning(f'found the argument "add_special_tokens" in "tokenizer_other_kwargs". change to giving the argument directly to the fucntion.')
            add_special_tokens = tokenizer_other_kwargs["add_special_tokens"]
        tokenizer_other_kwargs["add_special_tokens"] = False

        if add_special_tokens and tokenizer_other_kwargs.get("truncation", False):
            assert "max_length" in tokenizer_other_kwargs
            tokenizer_other_kwargs["max_length"] = tokenizer_other_kwargs["max_length"] - __TWO_BECAUSE_OF_SPECIAL_TOKEN

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
                # else: # comment out because this must never happens
                #     continue

                token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))

            else:
                st = start_to_token_id.get(span.s, None)
                et_minus_one = end_to_token_id.get(span.e, None)
                if (st is not None) and (et_minus_one is not None):
                    token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))
        self.token_spans = token_spans

        if add_special_tokens:
            self.add_special_tokens_(tokenizer=tokenizer)

        return self

    def add_special_tokens_(self, tokenizer:_transformers.PreTrainedTokenizer):
        assert not self.is_added_special_tokens, f'already special tokens are added. id:{self.id}'

        token_len_wo_sp_tokens = len(self.input_ids)
        new_input_ids = tokenizer.build_inputs_with_special_tokens(self.input_ids)
        assert len(new_input_ids) == token_len_wo_sp_tokens + __TWO_BECAUSE_OF_SPECIAL_TOKEN

        last_position = self.offset_mapping_end[-1]
        new_offset_mapping_start = [0] + self.offset_mapping_start + [last_position]
        new_offset_mapping_end = [0] + self.offset_mapping_end + [last_position]

        new_token_spans = list()
        for span in self.token_spans:
            copied_span = NERSpan(**_D.asdict(span))
            copied_span.s += (__TWO_BECAUSE_OF_SPECIAL_TOKEN // 2)
            copied_span.e += (__TWO_BECAUSE_OF_SPECIAL_TOKEN // 2)
            new_token_spans.append(copied_span)

        self.input_ids = new_input_ids
        self.offset_mapping_start = new_offset_mapping_start
        self.offset_mapping_end = new_offset_mapping_end
        self.token_spans = new_token_spans
        self.is_added_special_tokens = True
        return self

    def get_sequence_label(self, tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SingleLabel, only_label:_Optional[int]=None, num_class_without_negative=None, strict:bool=True) -> _Union[_List[int],_List[_List[int]]]:
        """
        output := [label_0, label_1, label_2, ...]

        if label_scheme==SingleLabel:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:L-class_0, 4:U-class_0, 5:B-class_1, 6:I-class_1, ..., 4*n+1:B-class_n, 4*n+2:I-class_n, 4*n+3:L-class_n, 4*n+4:U-class_n, ...}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B-class_0, 2:I-class_0, 3:B-class_1, 4:I-class_1, ..., 2*n+1:B-class_n, 2*n+2:I-class_n, ...}
            if tagging_scheme==Independent:
                label_t \in {0:O, 1:class_0, 2:class_1, ..., n+1:class_n, ...}

        if label_scheme==MultiLabel:
            label_t := [label_t_class_0, label_t_class_1, ..., label_t_class_N]
            N: num_class_without_negative

            label_t_class_k := int
            k: the index of the class

            if tagging_scheme==BILOU:
                label_t_class_k \in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t_class_k \in {0:O, 1:B, 2:I}
            if tagging_scheme==Independent:
                label_t_class_k \in {0:Negative, 1:Positive}

        if label_scheme==SpanOnly:
            label_t := int
            t: timestep

            if tagging_scheme==BILOU:
                label_t \in {0:O, 1:B, 2:I, 3:L, 4:U}
            if tagging_scheme==BIO:
                label_t \in {0:O, 1:B, 2:I}
            if tagging_scheme==Independent:
                label_t \in {0:Negative, 1:Positive}
        """
        if label_scheme == NERLabelScheme.MultiLabel:
            assert num_class_without_negative is not None, "num_class_without_negative must be specified under the multi-labelling setting."
        if only_label is None:
            target_spans = self.token_spans
        else:
            target_spans = [span for span in self.token_spans if span.l == only_label]

        if label_scheme in [NERLabelScheme.SingleLabel, NERLabelScheme.SpanOnly]:
            out = [int(NERSpanTag.O) for _ in range(len(self.input_ids))]
        elif label_scheme == NERLabelScheme.MultiLabel:
            transposed_out = [[int(NERSpanTag.O) for _ in range(len(self.input_ids))] for _ in range(num_class_without_negative)]
        else:
            raise ValueError(label_scheme)

        for span in target_spans:
            if tagging_scheme == NERTaggingScheme.BILOU:
                if label_scheme == NERLabelScheme.SingleLabel:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                    target_out = out
                    class_offset = span.l * 4 # 4 <- len({B, I, L, U})
                elif label_scheme == NERLabelScheme.SpanOnly:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MultiLabel:
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
                if label_scheme == NERLabelScheme.SingleLabel:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                    target_out = out
                    class_offset = span.l * 2 # 2 <- len({B, I})
                elif label_scheme == NERLabelScheme.SpanOnly:
                    if strict:
                        assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MultiLabel:
                    target_out = transposed_out[span.l]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                target_out[span.s] = NERSpanTag.B + class_offset
                for i in range(span.s+1, span.e):
                    target_out[i] = NERSpanTag.I + class_offset

            elif tagging_scheme == NERTaggingScheme.Independent:
                if label_scheme == NERLabelScheme.SingleLabel:
                    target_out = out
                    class_offset = span.l
                elif label_scheme == NERLabelScheme.SpanOnly:
                    target_out = out
                    class_offset = 0
                elif label_scheme == NERLabelScheme.MultiLabel:
                    target_out = transposed_out[span.l]
                    class_offset = 0
                else:
                    raise ValueError(label_scheme)

                for i in range(span.s, span.e):
                    target_out[i] = 1 + class_offset

            else:
                raise ValueError(tagging_scheme)

        if label_scheme == NERLabelScheme.MultiLabel:
            out = [list(v) for v in zip(*transposed_out)]

        return out

    def decode_token_span_to_char_span(self, span:_Union[NERSpan, _List[NERSpan]], strip=False) -> _Union[NERSpan, _List[NERSpan]]:
        if not isinstance(span, NERSpan):
            return [self.decode_token_span_to_char_span(s, strip=strip) for s in span]

        char_start = self.offset_mapping_start[span.s]
        if span.s == span.e:
            char_end = char_start
        else:
            char_end = self.offset_mapping_end[span.e-1]

        if strip:
            while (char_start < char_end) and (self.text[char_start] == " "):
                char_start += 1
            while (char_start < char_end) and (self.text[char_end-1] == " "):
                char_end -= 1

        return NERSpan(s=char_start, e=char_end, l=span.l, id=span.id)


# %%
def convert_sequence_label_to_spans(sequence_label:_Union[_List[int],_List[_List[int]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SingleLabel) -> _List[NERSpan]:
    if label_scheme in [NERLabelScheme.SingleLabel, NERLabelScheme.SpanOnly]:
        if tagging_scheme == NERTaggingScheme.Independent:
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

    elif label_scheme == NERLabelScheme.MultiLabel:
        num_class = len(sequence_label[0])
        outs = list()
        for c in range(num_class):
            sequence_label_class_c = [labels[c] for labels in sequence_label]
            spans_class_c = convert_sequence_label_to_spans(sequence_label=sequence_label_class_c, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SpanOnly)
            outs.extend([NERSpan(s=span.s, e=span.e, l=c) for span in spans_class_c])
        return outs

    else:
        raise ValueError(label_scheme)

def viterbi_decode(logits_sequence:_Union[_List[float],_List[_List[float]],_List[_List[_List[float]]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SingleLabel, scalar_logit_for_independent:bool=False) -> _Union[_List[int],_List[_List[int]]]:
    """
    input: logits_sequence
    - 2D or 3D float list. shape==[seq_len, [num_class,] num_label].

    output: sequence_label

    if (scalar_logit_for_independent == True) and (tagging_scheme == Independent) and (label_scheme in [MultiLabel, SpanOnly]),
    the expected shape of logits_sequence is [seq_len, [num_class,]] and each value logits_sequence[i[,j]] will be treated as a logit of the positive probability.
    otherwise, the shape should be [seq_len, [num_class,] 2] (2:(logit_negative, logit_positive)).


    if label_scheme==SingleLabel:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_L-class_0, logit_U-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*4 + 1 (4 <= {B,I,L,U})

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B-class_0, logit_I-class_0, logit_B-class_1, logit_I-class_1, ...]
            len(logits_time-step_t) == num_class*2 + 1 (2 <= {B,I})

        if tagging_scheme==Independent
            logits_time-step_t := [logit_O, logit_class_0, logit_class_1, ...]
            len(logits_time-step_t) == num_class*1 + 1 (1 <= {Positive})


    if label_scheme==MultiLabel:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]
        logits_time-step_t := [logits_class_0, logits_class_1, ...]

        if tagging_scheme==BILOU:
            logits_class_c := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_class_c := [logit_O, logit_B, logit_I]

        if tagging_scheme==Independent
            logits_class_c := [logit_Negative, logit_Positive]


    if label_scheme==SpanOnly:
        logits_sequence := [logits_time-step_0, logits_time-step_1, ...]

        if tagging_scheme==BILOU:
            logits_time-step_t := [logit_O, logit_B, logit_I, logit_L, logit_U]

        if tagging_scheme==BIO
            logits_time-step_t := [logit_O, logit_B, logit_I]

        if tagging_scheme==Independent
            logits_time-step_t := [logit_Negative, logit_Positive]
    """

    logits_sequence:_numpy.ndarray = _numpy.array(logits_sequence, dtype=_numpy.float32)

    if label_scheme == NERLabelScheme.SingleLabel:
        assert len(logits_sequence.shape) == 2, logits_sequence.shape

        if tagging_scheme == NERTaggingScheme.BILOU:
            num_class = (logits_sequence.shape[1] - 1) // 4
        elif tagging_scheme == NERTaggingScheme.BIO:
            num_class = (logits_sequence.shape[1] - 1) // 2
        elif tagging_scheme == NERTaggingScheme.Independent:
            num_class = logits_sequence.shape[1] - 1
        else:
            raise ValueError(tagging_scheme)
    elif label_scheme == NERLabelScheme.MultiLabel:
        if (tagging_scheme == NERTaggingScheme.Independent) and scalar_logit_for_independent:
            assert len(logits_sequence.shape) == 2, logits_sequence.shape
        else:
            assert len(logits_sequence.shape) == 3, logits_sequence.shape

        decoded = list()
        for class_i in range(logits_sequence.shape[1]):
            logit_sequence_class_i = logits_sequence[:,class_i]
            decoded_class_i = viterbi_decode(logits_sequence=logit_sequence_class_i, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SpanOnly, scalar_logit_for_independent=scalar_logit_for_independent)
            decoded.append(decoded_class_i)
        decoded = list(zip(*decoded)) # [num_class, seq_len] -> [seq_len, num_class]
        return decoded
    elif label_scheme == NERLabelScheme.SpanOnly:
        if (tagging_scheme == NERTaggingScheme.Independent) and scalar_logit_for_independent:
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

    elif tagging_scheme == NERTaggingScheme.Independent:
        if label_scheme == NERLabelScheme.SpanOnly and scalar_logit_for_independent:
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
        add_special_tokens=True,
    )
    print("instance:", instance)
    print()

    # %%
    print("multi class single labelling sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SingleLabel))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SingleLabel))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SingleLabel))
    print()

    # %%
    print("multi labelling sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2))
    print()

    # %%
    print("span-only (no-class) sequence label:")
    print("BILOU ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly))
    print("BILOU only for class_0 ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly, only_label=0))
    print("BIO ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SpanOnly))
    print("token-level ->", instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SpanOnly))
    print()

    # %%
    print(instance.token_spans)
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SingleLabel))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SingleLabel))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SingleLabel))

    # %%
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MultiLabel))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MultiLabel))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel))

    # %%
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SpanOnly))
    print(convert_sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SpanOnly))

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

    decoded1 = viterbi_decode(logits, tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel)
    decoded2 = viterbi_decode(positive_logits, tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel, scalar_logit_for_independent=True)
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
