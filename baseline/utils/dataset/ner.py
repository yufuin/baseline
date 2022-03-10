# %%
import dataclasses as _D
import enum as _enum
from typing import List as _List, Optional as _Optional, Tuple as _Tuple, Union as _Union, Any as _Any

import transformers as _transformers

# %%
@_D.dataclass
class NERSpan:
    s: int
    e: int
    l: int
    id: _Any = None
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

    input_ids: _Optional[_List[int]] = None
    offset_mapping_start: _Optional[_List[int]] = None
    offset_mapping_end: _Optional[_List[int]] = None
    token_spans: _Optional[_List[NERSpan]] = None

    @classmethod
    def build(cls, text:str, spans:_List[_Union[NERSpan,NERSpanAsList]], check_some:bool=True, tokenizer:_Optional[_transformers.PreTrainedTokenizer]=None, fuzzy=None):
        spans = [span if type(span) is NERSpan else NERSpan(*span) for span in spans]
        out = cls(text=text, spans=spans)
        if tokenizer is not None:
            encode_func_args = dict()
            if fuzzy is not None:
                encode_func_args["fuzzy"] = fuzzy
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

    def encode_(self, tokenizer, fuzzy=True):
        enc = tokenizer(self.text, return_offsets_mapping=True, add_special_tokens=False)
        self.input_ids = enc["input_ids"]
        self.offset_mapping_start = [se[0] for se in enc["offset_mapping"]]
        self.offset_mapping_end = [se[1] for se in enc["offset_mapping"]]

        start_to_token_id = {t:i for i,t in enumerate(self.offset_mapping_start)}
        end_to_token_id = {t:i for i,t in enumerate(self.offset_mapping_end)}

        token_spans = list()
        offset_mapping_end_with_sentinel = [0] + list(self.offset_mapping_end)
        offset_mapping_start_with_sentinel = list(self.offset_mapping_start) + [len(self.text)]
        for span in self.spans:
            if fuzzy:
                for st in range(len(self.input_ids)):
                    if offset_mapping_end_with_sentinel[st] <= span.s < offset_mapping_end_with_sentinel[st+1]:
                        break
                else:
                    continue

                for et_minus_one in range(len(self.input_ids)):
                    if offset_mapping_start_with_sentinel[et_minus_one] < span.e <= offset_mapping_start_with_sentinel[et_minus_one+1]:
                        break
                else:
                    continue

                token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))

            else:
                st = start_to_token_id.get(span.s, None)
                et_minus_one = end_to_token_id.get(span.e, None)
                if (st is not None) and (et_minus_one is not None):
                    token_spans.append(NERSpan(s=st,e=et_minus_one+1, l=span.l, id=span.id))
        self.token_spans = token_spans
        return self

    def sequence_tag(self, scheme:NERTaggingScheme=NERTaggingScheme.BILOU, no_class:bool=False, target_label:_Optional[int]=None, strict:bool=True) -> _List[int]:
        if target_label is None:
            target_spans = self.token_spans
        else:
            target_spans = [span for span in self.token_spans if span.l == target_label]

        out = [int(NERSpanTag.O) for _ in range(len(self.input_ids))]
        for span in target_spans:
            if scheme == NERTaggingScheme.BILOU:
                if strict:
                    assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                class_offset = 0 if no_class else span.l * 4 # 4 <- len({B, I, L, U})
                if span.s + 1 == span.e:
                    out[span.s] = NERSpanTag.U + class_offset
                else:
                    out[span.s] = NERSpanTag.B + class_offset
                    out[span.e-1] = NERSpanTag.L + class_offset
                    for i in range(span.s+1, span.e-1):
                        out[i] = NERSpanTag.I + class_offset
            elif scheme == NERTaggingScheme.BIO:
                if strict:
                    assert all([t == NERSpanTag.O for t in out[span.s:span.e]]), span
                class_offset = 0 if no_class else span.l * 2 # 2 <- len({B, I})
                out[span.s] = NERSpanTag.B + class_offset
                for i in range(span.s+1, span.e):
                    out[i] = NERSpanTag.I + class_offset
            elif scheme == NERTaggingScheme.Independent:
                class_offset = 0 if no_class else span.l
                for i in range(span.s, span.e):
                    out[i] = 1 + class_offset

            else:
                raise ValueError(scheme)
        return out

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
    print("multi class sequence tag:")
    print("BILOU ->", instance.sequence_tag(scheme=NERTaggingScheme.BILOU, no_class=False))
    print("BIO ->", instance.sequence_tag(scheme=NERTaggingScheme.BIO, no_class=False))
    print("token-level ->", instance.sequence_tag(scheme=NERTaggingScheme.Independent, no_class=False))
    print()

    # %%
    print("single class sequence tag:")
    print("BILOU ->", instance.sequence_tag(scheme=NERTaggingScheme.BILOU, no_class=True))
    print("BILOU only for class_0 ->", instance.sequence_tag(scheme=NERTaggingScheme.BILOU, no_class=True, target_label=0))
    print("BIO ->", instance.sequence_tag(scheme=NERTaggingScheme.BIO, no_class=True))
    print("token-level ->", instance.sequence_tag(scheme=NERTaggingScheme.Independent, no_class=True))
    print()

# %%
