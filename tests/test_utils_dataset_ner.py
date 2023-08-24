import unittest

import pydantic

import numpy as np
import transformers

import baseline.utils.dataset.ner as ner
ner.change_logging_level("ERROR")

class NERSpanTestCase(unittest.TestCase):
    def test_load_from_dict(self):
        span = ner.NERSpan(start=3, end=7, label=2, id="foo")
        dumped = span.model_dump_json()
        self.assertEqual(span, ner.NERSpan.model_validate_json(dumped))

    def test_check_some(self):
        span1 = ner.NERSpan(start=3, end=7, label=0)

        with self.assertRaises(pydantic.ValidationError):
            span2 = ner.NERSpan(start=7, end=3, label=0)

        # NOTE: currently accept zero-size span.
        # with self.assertRaises(pydantic.ValidationError):
        #     span3 = ner.NERSpan(start=7, end=7, label=0)

    def test_set(self):
        span_set1 = {
            ner.NERSpan(start=2, end=5, label=0, id="foo"),
            ner.NERSpan(start=3, end=8, label=1, id="bar")
        }
        span_set2 = {
            ner.NERSpan(start=2, end=5, label=0, id="piyo"),
            ner.NERSpan(start=3, end=8, label=1, id="piyopiyo")
        }
        self.assertNotEqual(span_set1, span_set2)

        span_set1_without_id = {span.without_id() for span in span_set1}
        span_set2_without_id = {span.without_id() for span in span_set2}
        self.assertEqual(span_set1_without_id, span_set2_without_id)

        span_set3 = {
            ner.NERSpan(start=2, end=5, label=0),
            ner.NERSpan(start=3, end=8, label=1)
        }
        self.assertEqual(span_set1_without_id, span_set3)

        span_set4 = {
            ner.NERSpan(start=2, end=5, label=0),
            ner.NERSpan(start=3, end=8, label=1),
            ner.NERSpan(start=24, end=42, label=1)
        }
        self.assertNotEqual(span_set1_without_id, span_set4)


class NERInstanceTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = transformers.AutoTokenizer.from_pretrained("roberta-base", trim_offsets=False)
        cls.tokenizer_with_trimming = transformers.AutoTokenizer.from_pretrained("roberta-base", trim_offsets=True)
    def setUp(self) -> None:
        pass
    def tearDown(self) -> None:
        pass

    def build_basic_instance(self, **otherargs) -> ner.NERInstance:
        # safe instance
        instance = ner.NERInstance.build(
            text = "This is an example.",
            spans = [{"start":0,"end":4, "label":0, "id":"first-span:class_0"}, {"start":5,"end":7, "label":1, "id":"second-span:class_1"}, {"start":8,"end":18, "label":0, "id":"third-span:class_0"}],
            **otherargs
        )
        return instance
    def build_unsafe_instance1(self, **otherargs) -> ner.NERInstance:
        # unsafe instance. spans begin/end in the middle of the word.
        instance = ner.NERInstance.build(
            text = "this is another example.",
            spans = [{"start":1,"end":4, "label":0, "id":'t"his"'}, {"start":6,"end":14, "label":1, "id":'i"s anothe"r'}, {"start":16,"end":22, "label":0, "id":'"exampl"e'}],
            **otherargs
        )
        return instance
    def build_unsafe_instance2(self, **otherargs) -> ner.NERInstance:
        # unsafe instance. spans begin/end on the blanks.
        instance = ner.NERInstance.build(
            text = "this   has so  many   blanks.",
            spans = [{"start":7,"end":19, "label":0, "id":'has so  many'}, {"start":4,"end":21, "label":0, "id":'   "has so  many"  '}],
            **otherargs
        )
        return instance
    def build_very_long_instance(self, **otherargs) -> ner.NERInstance:
        instance = ner.NERInstance.build(
            text = "there is very long text in this instance. we need to truncate this instance so that the bert model can take this as the input.",
            spans = [{"start":9,"end":9+14, "label":0, "id":"very long text"}, {"start":88,"end":88+10, "label":3, "id":"the bert model"}, {"start":116,"end":116+9, "label":2, "id":"the input"}],
            **otherargs
        )
        return instance

    def test_trimed_tokenizer(self):
        with self.assertRaises(ValueError):
            instance1 = self.build_basic_instance(tokenizer=self.tokenizer_with_trimming)

        instance2 = self.build_basic_instance(tokenizer=self.tokenizer)

    def test_build_and_load(self):
        instance = self.build_basic_instance()
        dumped1_1 = instance.model_dump()
        loaded1_1 = ner.NERInstance.model_validate(dumped1_1)
        self.assertEqual(instance, loaded1_1)
        dumped1_2 = instance.model_dump_json()
        loaded1_2 = ner.NERInstance.model_validate_json(dumped1_2)
        self.assertEqual(instance, loaded1_2)

        instance.encode_(tokenizer=self.tokenizer)
        dumped2 = instance.model_dump_json()
        loaded2 = ner.NERInstance.model_validate_json(dumped2)
        self.assertNotEqual(instance, loaded1_1)
        self.assertEqual(instance, loaded2)

    def test_encode(self):
        instance1 = self.build_basic_instance()
        ret_value = instance1.encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MAXIMIZE)
        self.assertIs(instance1, ret_value)
        self.assertNotEqual({span.without_id() for span in instance1.spans}, {span.without_id() for span in instance1.token_spans})
        self.assertEqual({span.without_id() for span in instance1.spans}, {span.without_id() for span in instance1.decode_token_span_to_char_span(instance1.token_spans, strip=True)})

        instance1_by_build = self.build_basic_instance(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MAXIMIZE)
        self.assertEqual(instance1.token_ids, instance1_by_build.token_ids)
        self.assertEqual(instance1.token_spans, instance1_by_build.token_spans)
        self.assertEqual(instance1.offset_mapping, instance1_by_build.offset_mapping)
        self.assertEqual(len(instance1.spans), len(instance1.token_spans))

        instance1_minimized_fitting = self.build_basic_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MINIMIZE)
        self.assertIsNot(instance1, instance1_minimized_fitting)
        self.assertEqual(instance1.token_ids, instance1_minimized_fitting.token_ids)
        self.assertNotEqual(instance1.token_spans, instance1_minimized_fitting.token_spans)
        self.assertNotEqual(len(instance1_minimized_fitting.spans), len(instance1_minimized_fitting.token_spans))

        instance1_minimized_fitting_with_trimming = self.build_basic_instance().encode_(tokenizer=self.tokenizer_with_trimming, ignore_trim_offsets=True, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MINIMIZE)
        self.assertIsNot(instance1, instance1_minimized_fitting_with_trimming)
        self.assertEqual(instance1.token_ids, instance1_minimized_fitting_with_trimming.token_ids)
        self.assertEqual(instance1.token_spans, instance1_minimized_fitting_with_trimming.token_spans)
        self.assertEqual(len(instance1_minimized_fitting_with_trimming.spans), len(instance1_minimized_fitting_with_trimming.token_spans))

        instance2 = self.build_unsafe_instance1().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MAXIMIZE)
        instance2_minimized_fitting = self.build_unsafe_instance1().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE, fit_token_span=ner.NERSpanFittingScheme.MINIMIZE)
        self.assertIsNot(instance2, instance2_minimized_fitting)
        self.assertEqual(instance2.token_ids, instance2_minimized_fitting.token_ids)
        self.assertNotEqual(instance2.token_spans, instance2_minimized_fitting.token_spans)
        self.assertEqual(len(instance2.spans), len(instance2.token_spans))
        self.assertNotEqual(len(instance2_minimized_fitting.spans), len(instance2_minimized_fitting.token_spans))

        ref_spans1 = {span.without_id() for span in instance1.spans}
        self.assertEqual(len(ref_spans1), len(instance1.spans))
        recovered_spans1 = {span.without_id() for span in instance1.decode_token_span_to_char_span(instance1.token_spans, strip=True)}
        self.assertEqual(ref_spans1, recovered_spans1)

        ref_spans2 = {span.without_id() for span in instance2.spans}
        self.assertEqual(len(ref_spans2), len(instance2.spans))
        recovered_spans1 = {span.without_id() for span in instance2.decode_token_span_to_char_span(instance2.token_spans, strip=True)}
        self.assertNotEqual(ref_spans2, recovered_spans1)

    def test_truncate_and_with_special_tokens(self):
        # with_special_tokens (no truncation)
        instance1 = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True, truncation=ner.NERTruncationScheme.NONE)
        instance1_without_sp = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.NONE)
        self.assertEqual(len(instance1.token_ids)-2, len(instance1_without_sp.token_ids))
        self.assertEqual(instance1.token_ids[1:-1], instance1_without_sp.token_ids)
        self.assertEqual(len(instance1.token_ids), len(instance1.offset_mapping))
        self.assertEqual(instance1.token_ids, [self.tokenizer.bos_token_id] + instance1_without_sp.token_ids + [self.tokenizer.eos_token_id])
        self.assertEqual([span.model_copy(update={"start":span.start-1, "end":span.end-1}) for span in instance1.token_spans], instance1_without_sp.token_spans)

        non_truncated_instance1_by_arg_False = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True, truncation=False)
        self.assertEqual(instance1.token_ids, non_truncated_instance1_by_arg_False.token_ids)
        self.assertEqual(instance1.token_spans, non_truncated_instance1_by_arg_False.token_spans)

        with self.assertRaises(AssertionError):
            instance = self.build_basic_instance()
            instance.encode_(tokenizer=self.tokenizer, add_special_tokens=True)
            instance.with_special_tokens_(tokenizer=self.tokenizer)
        instance1_forced_with_sp = instance1_without_sp.with_special_tokens(tokenizer=self.tokenizer)
        self.assertIsNot(instance1_forced_with_sp, instance1_without_sp)
        self.assertNotEqual(instance1_without_sp.token_ids, instance1_forced_with_sp.token_ids)
        self.assertEqual(instance1.token_ids, instance1_forced_with_sp.token_ids)
        self.assertEqual(instance1.token_spans, instance1_forced_with_sp.token_spans)
        self.assertEqual(instance1.offset_mapping, instance1_forced_with_sp.offset_mapping)

        # truncate
        with self.assertRaises(AssertionError):
            # no arg for max_length
            self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True, truncation=ner.NERTruncationScheme.TRUNCATE)
        truncated_instance1 = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True, truncation=ner.NERTruncationScheme.TRUNCATE, max_length=8)
        self.assertNotEqual(instance1.token_ids, truncated_instance1.token_ids)
        self.assertNotEqual(len(truncated_instance1.spans), len(truncated_instance1.token_spans))
        self.assertNotEqual(instance1.token_spans, truncated_instance1.token_spans)
        self.assertNotEqual(instance1.token_ids[:8], truncated_instance1.token_ids)
        self.assertEqual(instance1.token_ids[:7] + [self.tokenizer.eos_token_id], truncated_instance1.token_ids)

        truncated_instance1_by_arg_True = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True, truncation=True, max_length=8)
        self.assertEqual(truncated_instance1.token_ids, truncated_instance1_by_arg_True.token_ids)
        self.assertEqual(truncated_instance1.token_spans, truncated_instance1_by_arg_True.token_spans)

        # split
        with self.assertRaises(AssertionError):
            # no arg for stride
            self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.SPLIT, max_length=8)
        with self.assertRaises(AssertionError):
            # no arg for max_length
            self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.SPLIT, stride=3)
        with self.assertRaises(AssertionError):
            # no arg for max_length and stride
            self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.SPLIT)
        split_instances1_without_sp = self.build_very_long_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=False, truncation=ner.NERTruncationScheme.SPLIT, max_length=8, stride=3)
        self.assertIs(type(split_instances1_without_sp), list)
        self.assertNotEqual({span.without_id() for span in instance1_without_sp.spans}, {span.without_id() for instance in split_instances1_without_sp for span in instance.spans})
        self.assertNotEqual({span.without_id() for span in instance1_without_sp.token_spans}, {span.without_id() for instance in split_instances1_without_sp for span in instance.token_spans})
        self.assertNotEqual({span.without_id() for span in instance1_without_sp.spans}, {span.without_id() for instance in split_instances1_without_sp for span in instance.decode_token_span_to_char_span(instance.token_spans, strip=True)})
        self.assertEqual({span.without_id() for span in instance1_without_sp.spans}, {span.without_id() for instance in split_instances1_without_sp for span in instance.recover_split_offset_of_char_spans(instance.decode_token_span_to_char_span(instance.token_spans, strip=True))})
        self.assertEqual({span.without_id() for span in instance1_without_sp.spans}, {span.without_id() for instance in split_instances1_without_sp for span in instance.decode_token_span_to_char_span(instance.token_spans, strip=True, recover_split=True)})

        # recover text
        recovered_text = [None for _ in range(len(instance1.text))]
        for instance in split_instances1_without_sp:
            start = instance.info.split.char_offset
            end = start + len(instance.text)
            for src,dest in enumerate(range(start, end)):
                if recovered_text[dest] is not None:
                    self.assertEqual(recovered_text[dest], instance.text[src])
                recovered_text[dest] = instance.text[src]
        self.assertEqual(instance1.text, "".join(recovered_text))

        # recover token_ids
        recovered_token_ids = [None for _ in range(len(instance1_without_sp.token_ids))]
        for instance in split_instances1_without_sp:
            start = instance.info.split.token_offset
            end = start + len(instance.token_ids)
            for src,dest in enumerate(range(start, end)):
                if recovered_token_ids[dest] is not None:
                    self.assertEqual(recovered_token_ids[dest], instance.token_ids[src])
                recovered_token_ids[dest] = instance.token_ids[src]
        self.assertEqual(list(instance1_without_sp.token_ids), recovered_token_ids)

        # with_query_and_special_tokens
        query = [42,43,44]
        with self.assertRaises(AssertionError):
            instance = self.build_very_long_instance()
            instance.encode_(tokenizer=self.tokenizer, add_special_tokens=True)
            # must be without sp
            instance.with_query_and_special_tokens_(tokenizer=self.tokenizer, encoded_query=query, max_length=10000000)
        instance1_with_query = instance1_without_sp.with_query_and_special_tokens(tokenizer=self.tokenizer, encoded_query=query, max_length=10000000)
        self.assertIsNot(instance1_with_query, instance1_without_sp)
        self.assertNotEqual(instance1_with_query.token_ids, instance1_without_sp.token_ids)
        self.assertNotEqual(instance1_with_query.token_ids, instance1.token_ids)
        self.assertNotEqual(instance1_with_query.token_spans, instance1_without_sp.token_spans)
        self.assertEqual(instance1_with_query.spans, instance1_without_sp.spans)
        self.assertEqual(instance1_with_query.spans, instance1_with_query.decode_token_span_to_char_span(instance1_with_query.token_spans, strip=True))


    def test_sequence_label(self):
        instance1 = self.build_basic_instance().encode_(tokenizer=self.tokenizer, add_special_tokens=True)
        num_classes = max(span.label for span in instance1.spans) + 1
        ref_spans1 = {span.without_id() for span in instance1.token_spans}
        ref_spans1_without_class = {span.model_copy(update={"label":0}) for span in ref_spans1}
        merged_ref_spans1_without_class = {span for span in ner.merge_spans(list(ref_spans1_without_class))}

        recovered_single_label_bilou_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SINGLE_LABEL), tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SINGLE_LABEL)
        recovered_single_label_bio_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.SINGLE_LABEL), tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.SINGLE_LABEL)
        recovered_single_label_indep_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.SINGLE_LABEL), tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.SINGLE_LABEL)
        self.assertEqual(ref_spans1, set(recovered_single_label_bilou_spans))
        self.assertEqual(ref_spans1, set(recovered_single_label_bio_spans))
        self.assertNotEqual(ref_spans1, set(recovered_single_label_indep_spans))
        self.assertEqual(ref_spans1, set(ner.merge_spans(recovered_single_label_indep_spans)))

        recovered_multi_label_bilou_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.MULTI_LABEL, num_class_without_negative=num_classes), tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.MULTI_LABEL)
        recovered_multi_label_bio_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.MULTI_LABEL, num_class_without_negative=num_classes), tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.MULTI_LABEL)
        recovered_multi_label_indep_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.MULTI_LABEL, num_class_without_negative=num_classes), tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.MULTI_LABEL)
        self.assertEqual(ref_spans1, set(recovered_multi_label_bilou_spans))
        self.assertEqual(ref_spans1, set(recovered_multi_label_bio_spans))
        self.assertNotEqual(ref_spans1, set(recovered_multi_label_indep_spans))
        self.assertEqual(ref_spans1, set(ner.merge_spans(recovered_multi_label_indep_spans)))

        recovered_span_only_bilou_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SPAN_ONLY), tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SPAN_ONLY)
        recovered_span_only_bio_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.SPAN_ONLY), tagging_scheme=ner.NERTaggingScheme.BIO, label_scheme=ner.NERLabelScheme.SPAN_ONLY)
        recovered_span_only_indep_spans = ner.convert_sequence_label_to_spans(instance1.get_sequence_label(tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.SPAN_ONLY), tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.SPAN_ONLY)
        self.assertNotEqual(ref_spans1, set(recovered_span_only_bilou_spans))
        self.assertEqual(ref_spans1_without_class, set(recovered_span_only_bilou_spans))
        self.assertNotEqual(ref_spans1, set(recovered_span_only_bio_spans))
        self.assertEqual(ref_spans1_without_class, set(recovered_span_only_bio_spans))
        self.assertNotEqual(ref_spans1, set(recovered_span_only_indep_spans))
        self.assertNotEqual(ref_spans1_without_class, set(recovered_span_only_indep_spans))
        self.assertNotEqual(ref_spans1, set(ner.merge_spans(recovered_span_only_indep_spans)))
        self.assertNotEqual(ref_spans1_without_class, set(ner.merge_spans(recovered_span_only_indep_spans)))
        self.assertEqual(merged_ref_spans1_without_class, set(ner.merge_spans(recovered_span_only_indep_spans)))

    def test_viterbi_decode(self):
        logits1 = np.array([
            [-0.12580859661102295, 0.53866497576236725, 0.204946088418364525, 0.311039067059755325, 0.04118518531322479, 0.209847092628479, -0.240921288728714, -0.026520986109972, -0.07723920792341232],
            [-1.0395761728286743, -1.139390230178833, 1.67435884475708, 0.682197630405426, -0.005922921001911163, 0.4142416715621948, -0.8619325160980225, -0.527134120464325, 0.5418972969055176],
            [0.9126020669937134, -0.8182739019393921, -1.1124005317687988, -0.31609854102134705, -0.5066011548042297, -0.5548560619354248, -0.5321623682975769, -0.0609733872115612, -1.052380084991455],
            [-1.1847517490386963, -0.2960182726383209, 0.17322547733783722, -0.9496498107910156, -0.7404254078865051, -0.41485345363616943, 0.5226980447769165, 0.3296244442462921, -0.5696917772293091],
            [-0.048488348722457886, -0.028185393661260605, 0.45448189973831177, 0.571286141872406, 0.3106667101383209, 1.649383544921875, 1.14023838937282562, 2.0034116366878151894, -0.7123101353645325],
            [-1.1311267614364624, -1.0243971347808838, -0.9930524230003357, -0.3362758159637451, -0.5516917109489441, 0.08782649040222168, 1.3966140151023865, 1.19338271021842957, 0.9406710863113403],
            [0.9672456979751587, 1.850893497467041, 1.0942933559417725, 0.9000891447067261, -0.6152650713920593, 0.15397143363952637, -0.4109809100627899, 0.003905489109456539, -0.378101634979248],
        ]) # [seq_len=7, dim=1+4*2]
        binary_logits1 = np.stack([-logits1, logits1], axis=-1) # [7, 11, 2(negative/positive)]
        probs1 = 1 / (1+np.exp(-binary_logits1))

        # as token level multi label
        decoded1 = ner.viterbi_decode(binary_logits1, tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.MULTI_LABEL)
        decoded2 = ner.viterbi_decode(logits1, tagging_scheme=ner.NERTaggingScheme.TOKEN_LEVEL, label_scheme=ner.NERLabelScheme.MULTI_LABEL, scalar_logit_for_token_level=True)
        self.assertEqual(decoded1, decoded2)

        # as single label bilou
        decoded3 = ner.viterbi_decode(logits1, tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SINGLE_LABEL)
        decoded3_as_span = ner.viterbi_decode(logits1, tagging_scheme=ner.NERTaggingScheme.BILOU, label_scheme=ner.NERLabelScheme.SINGLE_LABEL, as_spans=True)
        self.assertEqual([1,3,0,5,7,8,0], list(decoded3))
        self.assertEqual({ner.NERSpan(start=0,end=2,label=0), ner.NERSpan(start=3,end=5,label=1), ner.NERSpan(start=5,end=6,label=1)}, set(decoded3_as_span))

        def dig(state:list, step, current_total_logit):
            if step == len(logits1):
                if state[-1][0] in ["O", "L", "U"]:
                    return [(current_total_logit, tuple(state))]
                else:
                    return []

            outs = list()
            past_state = state[-1] if len(state) > 0 else "O"
            for i,logit in enumerate(logits1[step]):
                next_logit = current_total_logit+logit
                next_step = step + 1
                class_ = (i-1)//4
                if i == 0: # "O"
                    if past_state[0] in ["O", "U", "L"]:
                        outs.extend(dig(state+["O"], next_step, next_logit))
                    continue
                elif (i-1) % 4 == 0: # "B"
                    if past_state[0] in ["O", "U", "L"]:
                        outs.extend(dig(state+[f"B{class_}"], next_step, next_logit))
                    continue
                elif (i-1) % 4 == 1: # "I"
                    if past_state in [f"B{class_}", f"I{class_}"]:
                        outs.extend(dig(state+[f"I{class_}"], next_step, next_logit))
                    continue
                elif (i-1) % 4 == 2: # "L"
                    if past_state in [f"B{class_}", f"I{class_}"]:
                        outs.extend(dig(state+[f"L{class_}"], next_step, next_logit))
                    continue
                elif (i-1) % 4 == 3: # "U"
                    if past_state[0] in ["O", "U", "L"]:
                        outs.extend(dig(state+[f"U{class_}"], next_step, next_logit))
                    continue
            return outs
        all_logits1 = dig(list(), 0, 0.0)
        max_logit1, max_code1 = max(all_logits1)
        i = 0
        max_span1 = set()
        while i < len(max_code1):
            if max_code1[i] == "O":
                pass
            elif max_code1[i][0] == "U":
                max_span1.add(ner.NERSpan(start=i,end=i+1,label=int(max_code1[i][1:])))
            elif max_code1[i][0] == "B":
                start = i
                while True:
                    i += 1
                    if max_code1[i][0] == "L":
                        break
                max_span1.add(ner.NERSpan(start=start,end=i+1,label=int(max_code1[i][1:])))
            i += 1
            continue
        self.assertEqual(max_span1, set(decoded3_as_span))

    def test_merge_spans(self):
        ref_spans = {
            ner.NERSpan(start=3,end=4, label=99),
            ner.NERSpan(start=10,end=15, label=101),
            ner.NERSpan(start=18,end=20, label=101),
            ner.NERSpan(start=85,end=90, label=100)
        }
        token_level_spans = [
            ner.NERSpan(start=3, end=4, label=99, id=None),
            ner.NERSpan(start=85, end=86, label=100, id=None),
            ner.NERSpan(start=86, end=87, label=100, id=None),
            ner.NERSpan(start=87, end=88, label=100, id=None),
            ner.NERSpan(start=88, end=89, label=100, id=None),
            ner.NERSpan(start=89, end=90, label=100, id=None),
            ner.NERSpan(start=10, end=11, label=101, id=None),
            ner.NERSpan(start=11, end=12, label=101, id=None),
            ner.NERSpan(start=12, end=13, label=101, id=None),
            ner.NERSpan(start=13, end=14, label=101, id=None),
            ner.NERSpan(start=14, end=15, label=101, id=None),
            ner.NERSpan(start=18, end=19, label=101, id=None),
            ner.NERSpan(start=19, end=20, label=101, id=None),
        ]
        rng = np.random.RandomState(123)
        rng.shuffle(token_level_spans)
        merged_spans = ner.merge_spans(token_level_spans)
        self.assertEqual(ref_spans, set(merged_spans))

    def test_strip_char_spans(self):
        instance1 = self.build_unsafe_instance2()
        ref_span = instance1.spans[0].without_id()
        base_span = instance1.spans[1].without_id()
        self.assertNotEqual(ref_span, base_span)
        stripped_span = instance1.strip_char_spans(base_span)
        self.assertEqual(ref_span, stripped_span)


