# %%
import logging as _logging
_logger = _logging.getLogger(__name__)
_logger.setLevel(_logging.WARNING)
_ch = _logging.StreamHandler()
_ch.setLevel(_logging.WARNING)
_formatter = _logging.Formatter('%(name)s - %(levelname)s:%(message)s')
_ch.setFormatter(_formatter)
_logger.addHandler(_ch)

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
    token_spans: _Optional[_List[NERSpan]] = None

    @classmethod
    def load_from_dict(cls, dumped):
        dumped = dict(dumped)
        dumped["spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["spans"]]
        if isinstance(dumped["token_spans"], list):
            dumped["token_spans"] = [NERSpan.load_from_dict(dic_span) for dic_span in dumped["token_spans"]]
        return cls(**dumped)

    @classmethod
    def build(cls, text:str, spans:_List[_Union[NERSpan,NERSpanAsList]], id:_Any=None, check_some:bool=True, tokenizer:_Optional[_transformers.PreTrainedTokenizer]=None, fuzzy:_Optional[bool]=None, tokenizer_kwargs:_Optional[dict]=None):
        spans = [span if type(span) is NERSpan else NERSpan(*span) for span in spans]
        out = cls(text=text, spans=spans, id=id)
        if tokenizer is not None:
            encode_func_args = dict()
            if fuzzy is not None:
                encode_func_args["fuzzy"] = fuzzy
            if tokenizer_kwargs is not None:
                encode_func_args["tokenizer_kwargs"] = tokenizer_kwargs
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

    def encode_(self, tokenizer:_transformers.PreTrainedTokenizer, fuzzy:bool=True, tokenizer_kwargs:_Optional[dict]=None):
        if tokenizer_kwargs is None:
            tokenizer_kwargs = dict()
        else:
            tokenizer_kwargs = dict(tokenizer_kwargs)
        if "add_special_tokens" not in tokenizer_kwargs:
            tokenizer_kwargs["add_special_tokens"] = False
        enc = tokenizer(self.text, return_offsets_mapping=True, **tokenizer_kwargs)
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
                # else:
                #     continue

                token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))

            else:
                st = start_to_token_id.get(span.s, None)
                et_minus_one = end_to_token_id.get(span.e, None)
                if (st is not None) and (et_minus_one is not None):
                    token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))
        self.token_spans = token_spans
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

# %%
def sequence_label_to_spans(sequence_label:_Union[_List[int],_List[_List[int]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SingleLabel) -> _List[NERSpan]:
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
            spans_class_c = sequence_label_to_spans(sequence_label=sequence_label_class_c, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SpanOnly)
            outs.extend([NERSpan(s=span.s, e=span.e, l=c) for span in spans_class_c])
        return outs

    else:
        raise ValueError(label_scheme)

def viterbi_decode(logits_sequence:_Union[_List[_List[int]],_List[_List[_List[int]]]], tagging_scheme:NERTaggingScheme=NERTaggingScheme.BILOU, label_scheme:NERLabelScheme=NERLabelScheme.SingleLabel) -> _Union[_List[int],_List[_List[int]]]:
    """
    input: logits_sequence
    - 2D or 3D float list. shape==[seq_len, [num_class,] num_label].

    output: sequence_label


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

    logits_sequence = _numpy.array(logits_sequence, dtype=_numpy.float32)

    if label_scheme == NERLabelScheme.SingleLabel:
        if tagging_scheme == NERTaggingScheme.BILOU:
            num_class = (logits_sequence.shape[1] - 1) // 4
        elif tagging_scheme == NERTaggingScheme.BIO:
            num_class = (logits_sequence.shape[1] - 1) // 2
        elif tagging_scheme == NERTaggingScheme.Independent:
            num_class = logits_sequence.shape[1] - 1
        else:
            raise ValueError(tagging_scheme)
    elif label_scheme == NERLabelScheme.MultiLabel:
        decoded = list()
        for class_i in range(logits_sequence.shape[1]):
            logit_sequence_class_i = logits_sequence[:,class_i]
            decoded_class_i = viterbi_decode(logits_sequence=logit_sequence_class_i, tagging_scheme=tagging_scheme, label_scheme=NERLabelScheme.SpanOnly)
            decoded.append(decoded_class_i)
        decoded = list(zip(*decoded)) # [num_class, seq_len] -> [seq_len, num_class]
        return decoded
    elif label_scheme == NERLabelScheme.SpanOnly:
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

# %%
if __name__ == "__main__":
    tok = _transformers.AutoTokenizer.from_pretrained("bert-base-cased")
    instance = NERInstance.build(
        text = "This is biomedical example.",
        spans = [[0,4, 0, "first-span:class_0"], [5,7, 1, "second-span:class_1"], (8,26, 0, "third-span:class_0")],
        tokenizer = tok
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
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SingleLabel))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SingleLabel))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SingleLabel), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SingleLabel))

    # %%
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.MultiLabel))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.MultiLabel))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel, num_class_without_negative=2), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.MultiLabel))

    # %%
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.BILOU, label_scheme=NERLabelScheme.SpanOnly))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.BIO, label_scheme=NERLabelScheme.SpanOnly))
    print(sequence_label_to_spans(instance.get_sequence_label(tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SpanOnly), tagging_scheme=NERTaggingScheme.Independent, label_scheme=NERLabelScheme.SpanOnly))

# %%
