import unittest

from baseline.utils import decoding

class BilouSpanDecodeTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_decode1(self):
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo"]), [("foo", (0,2))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "o", "o"]), [("foo", (0,2))])
        self.assertListEqual(decoding.bilou_span_decode(["o", "b-foo", "i-foo", "o", "o"]), [("foo", (1,3))])

        self.assertListEqual(decoding.bilou_span_decode(["o", "b-foo", "i-foo", "l-foo", "o"]), [("foo", (1,4))])
        self.assertListEqual(decoding.bilou_span_decode(["o", "b-foo", "i-foo", "i-foo", "l-foo", "o"]), [("foo", (1,5))])

        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "i-foo", "o", "i-foo"]), [("foo", (0,3)), ("foo", (4,5))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "i-foo", "o", "l-foo"]), [("foo", (0,3)), ("foo", (4,5))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "o", "i-foo", "i-foo", "i-foo"]), [("foo", (0,1)), ("foo", (2,5))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "o", "i-foo", "l-foo"]), [("foo", (0,2)), ("foo", (3,5))])

        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "b-bar", "i-bar"]), [("foo", (0,3)), ("bar", (3,5))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "b-bar", "l-bar"]), [("foo", (0,3)), ("bar", (3,5))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "o", "b-bar", "i-bar"]), [("foo", (0,3)), ("bar", (4,6))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "o", "b-bar", "l-bar"]), [("foo", (0,3)), ("bar", (4,6))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "b-bar", "o", "i-bar"]), [("foo", (0,3)), ("bar", (3,4)), ("bar", (5,6))])
        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "l-foo", "b-bar", "o", "l-bar"]), [("foo", (0,3)), ("bar", (3,4)), ("bar", (5,6))])

        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "i-foo", "i-bar", "l-bar"]), [("foo", (0,3)), ("bar", (3,5))])

        self.assertListEqual(decoding.bilou_span_decode(["b-foo", "i-foo", "o", "i-foo", "i-bar", "u-bar", "i-bar", "l-baz"]), [("foo", (0,2)), ("foo", (3,4)), ("bar", (4,5)), ("bar", (5,6)), ("bar", (6,7)), ("baz", (7,8)), ])


