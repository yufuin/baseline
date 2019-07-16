import unittest

from baseline.utils import padding as P

class PadTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test(self):
        seq = [[1, 2], [3,4,5,6], [7]]
        shape = P.get_padded_shape(seq)
        self.assertEqual(shape, [3, 4])
        padded, mask = P.pad(seq, padded_value=-1)
        self.assertEqual(padded, [[1,2,-1,-1], [3,4,5,6], [7,-1,-1,-1]])
        self.assertEqual(mask, [[1,1,0,0],[1,1,1,1],[1,0,0,0]])


