import unittest

from baseline.utils import dictionary as D

class DictionaryTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize(self):
        unk_symbol = "$$UNK$$"
        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)

        self.assertEqual(len(dictionary), 1)

        self.assertTrue(unk_symbol in dictionary.dic)
        self.assertEqual(dictionary.get(unk_symbol), dictionary.unk_symbol_id)
        self.assertEqual(dictionary.decode(dictionary.unk_symbol_id), unk_symbol)

    def test_dump_load(self):
        unk_symbol1 = "$$UNK-1$$"
        dictionary1 = D.BasicDictionary(unk_symbol=unk_symbol1)
        token1_id = dictionary1.add("token1")
        token2_id = dictionary1.add("token2")
        self.assertNotEqual(token1_id, token2_id)
        self.assertNotEqual(dictionary1.unk_symbol_id, token1_id)
        self.assertNotEqual(dictionary1.unk_symbol_id, token2_id)
        dump = dictionary1.dump()

        unk_symbol2 = "$$UNK-2$$"
        dictionary2 = D.BasicDictionary(unk_symbol=unk_symbol2)
        none_id = dictionary2.get("token1")
        self.assertEqual(none_id, dictionary2.unk_symbol_id)
        none_id = dictionary2.get("token2")
        self.assertEqual(none_id, dictionary2.unk_symbol_id)

        dictionary2.load(dump)
        reload_token1_id = dictionary2.get("token1")
        self.assertEqual(reload_token1_id, token1_id)
        reload_token2_id = dictionary2.get("token2")
        self.assertEqual(reload_token2_id, token2_id)
        self.assertEqual(dictionary1.unk_symbol_id, dictionary2.unk_symbol_id)

    def test_add_get_encode(self):
        unk_symbol = "$$UNK$$"

        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)
        repr_set1 = ["foo", "bar", 42]
        add_ids1 = dictionary.add(repr_set1)
        encode_ids1 = dictionary.encode(repr_set1, allow_new=True)
        self.assertEqual(add_ids1, encode_ids1)
        # reverse order
        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)
        repr_set1 = ["foo", "bar", 42]
        encode_ids1 = dictionary.encode(repr_set1, allow_new=True)
        add_ids1 = dictionary.add(repr_set1)
        self.assertEqual(add_ids1, encode_ids1)

        repr_set2 = ["foo", "foobar"]
        encode_ids2 = dictionary.encode(repr_set2, allow_new=False)
        self.assertTrue("foobar" not in dictionary.dic)
        self.assertEqual(encode_ids2, [1, dictionary.unk_symbol_id])
        get_ids2 = dictionary.get(repr_set2)
        self.assertTrue("foobar" not in dictionary.dic)
        self.assertEqual(get_ids2, encode_ids2)
        add_ids2 = dictionary.add(repr_set2)
        self.assertEqual(add_ids2, [1, 4])
        get_ids2_2 = dictionary.get(repr_set2)
        self.assertEqual(add_ids2, get_ids2_2)
        encode_ids2_2 = dictionary.encode(repr_set2, allow_new=False)
        self.assertEqual(get_ids2_2, encode_ids2_2)
        # reverse order
        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)
        dictionary.encode(repr_set1, allow_new=True)
        get_ids2 = dictionary.get(repr_set2)
        self.assertTrue("foobar" not in dictionary.dic)
        self.assertEqual(get_ids2, encode_ids2)
        encode_ids2 = dictionary.encode(repr_set2, allow_new=False)
        self.assertTrue("foobar" not in dictionary.dic)
        self.assertEqual(encode_ids2, [1, dictionary.unk_symbol_id])
        self.assertEqual(get_ids2_2, encode_ids2_2)

    def test_encode_and_decode(self):
        unk_symbol = "$$UNK$$"
        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)

        repr_set1 = ["foo", "bar", 42]
        ids1 = dictionary.encode(repr_set1, allow_new=True)
        self.assertEqual(ids1, [1, 2, 3])
        self.assertEqual(dictionary.decode(ids1), repr_set1)
        repr_set2 = ["foo", "foobar"]
        ids2 = dictionary.encode(repr_set2, allow_new=False)
        self.assertEqual(ids2, [1, dictionary.unk_symbol_id])

        self.assertEqual(dictionary.decode(ids1), repr_set1)
        self.assertEqual(dictionary.decode(ids2), ["foo", dictionary.unk_symbol])
        self.assertEqual(dictionary.decode(ids1), repr_set1)

    def test_len(self):
        unk_symbol = "$$UNK$$"
        dictionary = D.BasicDictionary(unk_symbol=unk_symbol)
        self.assertEqual(len(dictionary), 1)

        repr_set = ["foo", "bar", 42, "foo"]
        dictionary.encode(repr_set, allow_new=False)
        self.assertEqual(len(dictionary), 1)

        dictionary.encode(repr_set, allow_new=True)
        self.assertEqual(len(dictionary), 4)





