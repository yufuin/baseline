import unittest

from baseline.utils import padding as P
import numpy as np

class PadTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_1(self):
        seq = [[1, 2], [3,4,5,6], [7]]
        shape = P.get_padded_shape(seq)
        self.assertEqual(shape, [3, 4])
        padded, mask = P.pad(seq, padding_value=-1)
        self.assertEqual(padded, [[1,2,-1,-1], [3,4,5,6], [7,-1,-1,-1]])
        self.assertEqual(mask, [[1,1,0,0],[1,1,1,1],[1,0,0,0]])

    def test_2(self):
        seq = [[[1, 2]], [[3,4,5],[6]], [[]]]
        shape = P.get_padded_shape(seq)
        self.assertEqual(shape, [3, 2, 3])
        padded, mask = P.pad(seq, padding_value=-1)
        self.assertEqual(padded, [[[1,2,-1],[-1,-1,-1]], [[3,4,5],[6,-1,-1]], [[-1,-1,-1],[-1,-1,-1]]])
        self.assertEqual(mask, [[[1,1,0],[0,0,0]], [[1,1,1],[1,0,0]], [[0,0,0],[0,0,0]]])

    def test_3(self):
        seq = [[[1, 2]], [[3,4,5],[6]], []]
        shape = P.get_padded_shape(seq)
        self.assertEqual(shape, [3, 2, 3])
        padded, mask = P.pad(seq, padding_value=-1)
        self.assertEqual(padded, [[[1,2,-1],[-1,-1,-1]], [[3,4,5],[6,-1,-1]], [[-1,-1,-1],[-1,-1,-1]]])
        self.assertEqual(mask, [[[1,1,0],[0,0,0]], [[1,1,1],[1,0,0]], [[0,0,0],[0,0,0]]])

    def test_4(self):
        p = -1
        v1 = [[[1], [2]]]
        v2 = [[[3,4],[5]],[[6]]]
        v3 = []
        vs = [v1,v2,v3]
        p1 = [[[1,p],[2,p]],[[p,p],[p,p]]]
        p2 = [[[3,4],[5,p]],[[6,p],[p,p]]]
        p3 = [[[p,p],[p,p]],[[p,p],[p,p]]]
        ps = [p1,p2,p3]
        m1 = [[[1,0],[1,0]],[[0,0],[0,0]]]
        m2 = [[[1,1],[1,0]],[[1,0],[0,0]]]
        m3 = [[[0,0],[0,0]],[[0,0],[0,0]]]
        ms = [m1,m2,m3]

        for c in [[0,1,2],[0,2,1],[1,0,2],[1,2,0],[2,0,1],[2,1,0]]:
            cv = [vs[j] for j in c]
            cp = [ps[j] for j in c]
            cm = [ms[j] for j in c]

            shape = P.get_padded_shape(cv)
            self.assertEqual(shape, [3, 2, 2, 2])

            padded, mask = P.pad(cv, padding_value=p)
            self.assertEqual(padded, cp)
            self.assertEqual(mask, cm)

    def convert_all_numpy_to_all_list(self, structure):
        if type(structure) in [tuple, list]:
            return [self.convert_all_numpy_to_all_list(val) for val in structure]
        else:
            assert type(structure) is np.ndarray, structure
            return structure.tolist()
    def test_convert_all_numpy_to_all_list(self):
        seq = [[1, 2], [3,4,5,6], [7]]
        seq_n = [np.array([1, 2]), np.array([3,4,5,6]), np.array([7])]
        self.assertEqual(seq, self.convert_all_numpy_to_all_list(seq_n))
    def compare_padded_numpy_to_padded_lists(self, seq_n, padding_value):
        seq = self.convert_all_numpy_to_all_list(seq_n)

        shape = P.get_padded_shape(seq)
        shape_n = P.get_padded_shape_numpy(seq_n)
        self.assertEqual(shape, shape_n)

        padded, mask = P.pad(seq, padding_value=padding_value)
        padded_n, mask_n = P.pad_numpy(seq_n, padding_value=padding_value)
        self.assertTrue((np.array(padded) == padded_n).all())
        self.assertTrue((np.array(mask) == mask_n).all())

    def test_n1(self):
        seq_n = [np.array([1, 2]), np.array([3,4,5,6]), np.array([7])]
        self.compare_padded_numpy_to_padded_lists(seq_n, padding_value=-1)

    def test_n2(self):
        seq_n = [np.array([[1, 2]]), [np.array([3,4,5]),np.array([6])], [[]]]
        self.compare_padded_numpy_to_padded_lists(seq_n, padding_value=0)

    def test_n2_2(self):
        seq_n = [np.array([[1, 2]]), [np.array([3,4,5]),np.array([6])], [np.array([], dtype=np.int64)]]
        self.compare_padded_numpy_to_padded_lists(seq_n, padding_value=-1)

    def test_n3(self):
        seq_n = [[np.array([1, 2])], [np.array([3,4,5]),np.array([6])], []]
        self.compare_padded_numpy_to_padded_lists(seq_n, padding_value=-1)

    def test_n4(self):
        seq_n = [[np.array([[1, 2, 3], [4,5,6]])], [np.array([[11,12,13]]),np.array([[21]])], []]
        self.compare_padded_numpy_to_padded_lists(seq_n, padding_value=-1)


