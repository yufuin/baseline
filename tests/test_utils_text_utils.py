import unittest

from baseline.utils import text_utils as T

class SplitLinesWithPositionsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_only_n_return(self):
        # the same result as string.splitlines if text doens't have any newline characters at the EOS.
        text = "fizz\nbuzz\nbuzz\nfizz\nfizz\nfizzfizz\n\nfizz\nbuzz"
        ret = T.splitlines_with_positions(text)
        self.assertListEqual([s[0] for s in ret], text.splitlines())
        self.assertEqual("\n".join(s[0] for s in ret), text)

        # check when text has one or two newline charcter(s) at the EOS
        # one "\n"
        text = "fizz\nbuzz\nbuzz\nfizz\nfizz\nfizzfizz\n\nfizz\nbuzz\n"
        ret = T.splitlines_with_positions(text)
        self.assertListEqual([s[0] for s in ret], text.splitlines())
        self.assertEqual("\n".join(s[0] for s in ret), text[:-1])
        # two "\n"
        text = "fizz\nbuzz\nbuzz\nfizz\nfizz\nfizzfizz\n\nfizz\nbuzz\n\n"
        ret = T.splitlines_with_positions(text)
        self.assertListEqual([s[0] for s in ret], text.splitlines())
        self.assertEqual("\n".join(s[0] for s in ret), text[:-1])

        # keepends
        text = "fizz\nbuzz\nbuzz\nfizz\nfizz\nfizzfizz\n\nfizz\nbuzz\n\n"
        ret = T.splitlines_with_positions(text, keepends=True)
        self.assertListEqual([s[0] for s in ret], text.splitlines(keepends=True))
        self.assertEqual(text, "".join(s[0] for s in ret))

        # consistency of keepends
        text = "fizz\nbuzz\nbuzz\nfizz\nfizz\nfizzfizz\n\nfizz\nbuzz\n\n"
        ret_notkeep = T.splitlines_with_positions(text)
        ret_keep = T.splitlines_with_positions(text, keepends=True)
        self.assertListEqual([s[1] for s in ret_notkeep], [s[1] for s in ret_keep])

    def test_rn_return(self):
        # for the situation that text has both "\r" and "\n"

        text = "fizz\nbuzz\nbuzz\r\nfizz\nfizz\nfizzfizz\n\r\nfizz\nbuzz\n\n"
        ret = T.splitlines_with_positions(text, keepends=True)
        self.assertListEqual([s[0] for s in ret], text.splitlines(keepends=True))
        self.assertEqual(text, "".join(s[0] for s in ret))

        text = "fizz\nbuzz\nbuzz\r\nfizz\nfizz\nfizzfizz\n\r\nfizz\nbuzz\n\n"
        ret_notkeep = T.splitlines_with_positions(text)
        ret_keep = T.splitlines_with_positions(text, keepends=True)
        self.assertListEqual([s[1] for s in ret_notkeep], [s[1] for s in ret_keep])


