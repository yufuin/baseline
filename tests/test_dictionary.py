import unittest

from baseline.utils import dictionary as D

class DictionaryTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_initialize(self):
        dictionary = D.Dictionary()

        origin_unk_repr = D.Dictionary._UNK_REPR
        self.assertEqual(origin_unk_repr, dictionary.unk_repr)

        self.assertEqual(len(dictionary), 1)
        self.assertTrue(dictionary.unk_repr in dictionary._repr_to_id)
        self.assertTrue(dictionary.unk_repr in dictionary._id_to_repr)

        unk_id = dictionary.to_id(dictionary.unk_repr, allow_new=False)
        self.assertEqual(unk_id, dictionary.unk_id)

        reverted_unk_repr = dictionary.to_repr(unk_id)
        self.assertEqual(reverted_unk_repr, dictionary.unk_repr)

    def test_dump_load(self):
        dictionary1 = D.Dictionary()
        token1_id = dictionary1.to_id("token1", allow_new=True)
        token2_id = dictionary1.to_id("token2", allow_new=True)
        self.assertNotEqual(token1_id, token2_id)
        self.assertNotEqual(dictionary1.unk_id, token1_id)
        self.assertNotEqual(dictionary1.unk_id, token2_id)
        dump = dictionary1.dump()

        dictionary2 = D.Dictionary()
        none_id = dictionary2.to_id("token1", allow_new=False)
        self.assertEqual(none_id, dictionary2.unk_id)
        none_id = dictionary2.to_id("token2", allow_new=False)
        self.assertEqual(none_id, dictionary2.unk_id)

        dictionary2.load(dump)
        reload_token1_id = dictionary2.to_id("token1", allow_new=False)
        self.assertEqual(reload_token1_id, token1_id)
        reload_token2_id = dictionary2.to_id("token2", allow_new=False)
        self.assertEqual(reload_token2_id, token2_id)
        self.assertEqual(dictionary1.unk_id, dictionary2.unk_id)

    def test_to_id_and_repr(self):
        dictionary = D.Dictionary()

        repr_set1 = ["foo", "bar", 42]
        ids = [dictionary.to_id(r, allow_new=True) for r in repr_set1]
        self.assertEqual(ids, [1, 2, 3])

        repr_set2 = ["foo", "foobar"]
        ids = [dictionary.to_id(r, allow_new=False) for r in repr_set2]
        self.assertEqual(ids, [1, dictionary.unk_id])

        self.assertEqual(dictionary.to_repr(dictionary.to_id("foo", allow_new=False)), "foo")
        self.assertEqual(dictionary.to_repr(dictionary.to_id("bar", allow_new=False)), "bar")
        self.assertEqual(dictionary.to_repr(dictionary.to_id("foobar", allow_new=False)), dictionary.unk_repr)
        self.assertNotEqual(dictionary.to_repr(dictionary.to_id("foobar", allow_new=False)), "foobar")

    def test_to_id_and_repr_recursive(self):
        dictionary = D.Dictionary()

        reprs = {"foo": ["token1", "token2", {"bar":"token1", "baz":"token1"}, ["token3", "token4"]],
                 "foobar": {"foofoo": "token5", "barbar": "token3"}}
        ids = dictionary.to_id_recursive(reprs, allow_new=True)

        token1_id = dictionary.to_id("token1", allow_new=False)
        self.assertNotEqual(token1_id, dictionary.unk_id)
        token2_id = dictionary.to_id("token2", allow_new=False)
        self.assertNotEqual(token2_id, dictionary.unk_id)
        token3_id = dictionary.to_id("token3", allow_new=False)
        self.assertNotEqual(token3_id, dictionary.unk_id)
        token4_id = dictionary.to_id("token4", allow_new=False)
        self.assertNotEqual(token4_id, dictionary.unk_id)
        token5_id = dictionary.to_id("token5", allow_new=False)
        self.assertNotEqual(token5_id, dictionary.unk_id)

        self.assertEqual(len(set([token1_id, token2_id, token3_id, token4_id, token5_id])), 5)

        ideal_ids = {"foo": [token1_id, token2_id, {"bar":token1_id, "baz":token1_id}, [token3_id, token4_id]],
                     "foobar": {"foofoo": token5_id, "barbar": token3_id}}
        self.assertEqual(ids, ideal_ids)

        self.assertEqual(reprs, dictionary.to_repr_recursive(ids))

    def test_len(self):
        dictionary = D.Dictionary()
        self.assertEqual(len(dictionary), 1)

        repr_set = ["foo", "bar", 42, "foo"]
        for r in repr_set:
            dictionary.to_id(r, allow_new=False)
        self.assertEqual(len(dictionary), 1)

        for r in repr_set:
            dictionary.to_id(r, allow_new=True)
        self.assertEqual(len(dictionary), 4)





