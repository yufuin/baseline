# %%
import unittest

import io
import pydantic

import baseline.utils.dataset.brat as BRAT

# %%
class BratTestCase(unittest.TestCase):
    def test_declare_span(self):
        span = BRAT.Span(start=1,end=10)

        with self.assertRaises(pydantic.ValidationError):
            span = BRAT.Span(start=3,end=2)
        with self.assertRaises(pydantic.ValidationError):
            span = BRAT.Span(start=4,end=4)

        span = BRAT.Span.model_validate({"start":3, "end":5})

    def test_declare_text_bound(self):
        span1 = BRAT.Span(start=1,end=10)

        # always stored as list
        text_bound = BRAT.TextBound(id="T1", type="ent1", spans=span1, reference_text="X"*(span1.end-span1.start))
        self.assertIsInstance(text_bound.spans, list)
        text_bound = BRAT.TextBound(id="T2", type="ent2", spans=[span1], reference_text="X"*(span1.end-span1.start))
        self.assertIsInstance(text_bound.spans, list)

        # must have at least one span
        with self.assertRaises(pydantic.ValidationError):
            text_bound = BRAT.TextBound(id="T3", type="ent3", spans=[])

        # can take multiple discontinuous spans
        text_bound = BRAT.TextBound(id="T4", type="ent4", spans=[{"start":3, "end":5}, {"start":30, "end":50}], reference_text=" ".join(["X"*(5-3), "Y"*(50-30)]))
        # must be in ascending order
        with self.assertRaises(pydantic.ValidationError):
            text_bound = BRAT.TextBound(id="T5", type="ent5", spans=[{"start":30, "end":50}, {"start":3, "end":5}], reference_text=" ".join(["Y"*(50-30), "X"*(5-3)]))

        # initial of tag id must be "T"
        with self.assertRaises(pydantic.ValidationError):
            text_bound = BRAT.TextBound(id="U1", type="ent4", spans=[span1], reference_text="X"*(span1.end-span1.start))

    def test_text_bound_value(self):
        span1 = BRAT.Span(start=1,end=10)
        text_bound1 = BRAT.TextBound(id="T1", type="ent1", spans=span1, reference_text="X"*(span1.end-span1.start))
        self.assertEqual(text_bound1.start, 1)
        self.assertEqual(text_bound1.end, 10)

        text_bound2 = BRAT.TextBound(id="T2", type="ent2", spans=[{"start":3, "end":5}, {"start":30, "end":50}], reference_text=" ".join(["X"*(5-3), "Y"*(50-30)]))
        self.assertEqual(text_bound2.start, 3)
        self.assertEqual(text_bound2.end, 50)

        # can access TextBound.spans[0] by TextBound.span iff the text bound contains exactly one span
        should_be_span1 = text_bound1.span
        self.assertEqual(span1, should_be_span1)
        with self.assertRaises(ValueError):
            cannot_do_this = text_bound2.span

    def test_declare_relation(self):
        rel1 = BRAT.Relation(id="R1", type="Rel1", args=[BRAT.Argument(role="Arg1", arg_id="T1"), {"role":"Arg2", "arg_id":"T2"}, {"role":"Arg3", "arg_id":"T3"}])

        # must be at least one argument
        with self.assertRaises(pydantic.ValidationError):
            rel2 = BRAT.Relation(id="R2", type="Rel2", args=[])

        # initial of tag id must be "R"
        with self.assertRaises(pydantic.ValidationError):
            rel3 = BRAT.Relation(id="A3", type="Rel3", args=[BRAT.Argument(role="Arg1", arg_id="T1"), {"role":"Arg2", "arg_id":"T2"}, {"role":"Arg3", "arg_id":"T3"}])

    def test_declare_event(self):
        event1 = BRAT.Event(id="E1", type="Event1", trigger_id="T0", args=[BRAT.Argument(role="Arg1", arg_id="T1"), {"role":"Arg2", "arg_id":"T2"}, {"role":"Arg3", "arg_id":"T3"}])

        # initial of tag id must be "R"
        with self.assertRaises(pydantic.ValidationError):
            event3 = BRAT.Event(id="A2", type="Event2", trigger_id="T0", args=[BRAT.Argument(role="Arg1", arg_id="T1"), {"role":"Arg2", "arg_id":"T2"}, {"role":"Arg3", "arg_id":"T3"}])

    def test_declare_attribute(self):
        # value: Union[bool, str]
        attr1 = BRAT.Attribute(id="A1", name="Negation", target_id="T1", value=True)
        attr2 = BRAT.Attribute(id="A2", name="Negation", target_id="T1", value=False)
        attr3 = BRAT.Attribute(id="A3", name="Confidence", target_id="T1", value="High")
        attr4 = BRAT.Attribute(id="M1", name="Confidence", target_id="T1", value="VeryLow")

        # initial of tag id must be "A" or "M"
        with self.assertRaises(pydantic.ValidationError):
            attr5 = BRAT.Attribute(id="P1", name="Confidence", target_id="T1", value="VeryLow")

    def test_declare_normalization(self):
        norm1 = BRAT.Normalization(id="N1", target_id="T2", resource_id="Wikipedia", entry_id="Equivalence")

        # initial of tag id must be "N"
        with self.assertRaises(pydantic.ValidationError):
            norm2 = BRAT.Normalization(id="A1", target_id="T2", resource_id="Wikipedia", entry_id="Equivalence")

    def test_declare_document(self):
        document1 = BRAT.Document(text="foobar")

        document2 = BRAT.Document(
            text = "This is an example.",
            text_bounds = [{"id":"T1", "type":"entity1", "span":{"start":0, "end":4}, "reference_text":"This"}, {"id":"T2", "type":"entity2", "spans":[{"start":11, "end":18}], "reference_text":"example"}, {"id":"T3", "type":"EqEvent", "spans":[{"start":5, "end":7}], "reference_text":"is"}],
            relations = [{"id":"R1", "type":"equivalent", "args":[{"role":"Arg1", "arg_id":"T1"}, {"role":"Arg2", "arg_id":"T2"}]}],
            events = [{"id":"E1",  "type":"equivalent", "trigger_id":"T3", "args":[{"role":"Arg1", "arg_id":"T1"}, {"role":"Arg2", "arg_id":"T2"}]}],
            attributes = [BRAT.Attribute(id="A1", name="Negation", target_id="T1", value=True)],
            normalizations=[BRAT.Normalization(id="N1", target_id="T2", resource_id="Wikipedia", entry_id="Equivalence")],
        )

    def get_sample_ann_text1(self):
        text = "This is an example."
        ann = "\n".join([
            "T1\tEnt1 0 4\tThis",
            "T2\tEnt2 11 18\texample",
            "T3\tIS-A-Event 5 7\tis",
            "R1\tIS-A-Rel Arg1:T1 Arg2:T2",
            "E1\tIS-A-Event:T3 Parent:T2 Child:T1",
            "N1\tReference T3 Wikipedia:Q21503252",
        ])
        return text, ann

    def test_parse_ann_file(self):
        text, ann = self.get_sample_ann_text1()
        ann_file = io.StringIO(ann)
        doc = BRAT.parse_ann_file(ann_file=ann_file)


# %%

# %%



