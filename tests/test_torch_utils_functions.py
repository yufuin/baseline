import unittest

import torch

from baseline.torch.utils import functions as F

class EluClipTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_elu_clip(self):
        # -16.0 <= v <= 16.0 => elu_clip(v) == v
        self.assertAlmostEqual(0.0, F.elu_clip(torch.tensor(0.0)))
        for v in [0.0, 1.0, 8.1, 14.4, 16.0]:
            self.assertAlmostEqual(v, F.elu_clip(torch.tensor(v), -16.0, 16.0), delta=v*1e-5)
            self.assertAlmostEqual(-v, F.elu_clip(torch.tensor(-v), -16.0, 16.0), delta=v*1e-5)

        # v < -16.0 => -17.0 < elu_clip(v, -16.0, 16.0) < -16.0
        # 16.0 < v => 16.0 < elu_clip(v, -16.0, 16.0) < 17.0
        for v in [16.5, 17.0, 20.0]:
            self.assertNotAlmostEqual(v, F.elu_clip(torch.tensor(v), -16.0, 16.0), delta=v*1e-5)
            self.assertLess(16.0, F.elu_clip(torch.tensor(v), -16.0, 16.0))
            self.assertLess(F.elu_clip(torch.tensor(v), -16.0, 16.0), 17.0)

            self.assertNotAlmostEqual(-v, F.elu_clip(torch.tensor(-v), -16.0, 16.0), delta=v*1e-5)
            self.assertLess(-17.0, F.elu_clip(torch.tensor(-v), -16.0, 16.0))
            self.assertLess(F.elu_clip(torch.tensor(-v), -16.0, 16.0), -16.0)

class FlattenTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_flatten(self):
        v = torch.arange(2*3*5*7).reshape(2,3,5,7)
        self.assertEqual(v.shape, (2,3,5,7))
        self.assertEqual(F.flatten(v, [0,1]).shape, (2*3,5,7))
        self.assertEqual(F.flatten(v, [1,2]).shape, (2,3*5,7))
        self.assertEqual(F.flatten(v, [1,2,3]).shape, (2,3*5*7))

class MaxPoolTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_max_pool(self):
        v = torch.arange(2*5*3).reshape(2,5,3).to(torch.float32)
        mask = torch.FloatTensor([[0.0, 1.0, 0.0, 1.0, 0.0]]).repeat(2,1) # [2,5]
        """
        [
            [[0,1,2],    [3,4,5],    [6,7,8],    [9,10,11],  [12,13,14]],
            [[15,16,17], [18,19,20], [21,22,23], [24,25,26], [27,28,29]],
        ]
        => [
            [[0,0,0],    [3,4,5], [0,0,0], [9,10,11],  [0,0,0]],
            [[0,0,0], [18,19,20], [0,0,0], [24,25,26], [0,0,0]],
        ]
        """
        self.assertTrue(F.max_pool(v, mask).isclose(torch.FloatTensor([[9.0, 10.0, 11.0], [24.0, 25.0, 26.0]])).all())
        self.assertTrue(F.max_pool(v, mask, 1).isclose(torch.FloatTensor([[9.0, 10.0, 11.0], [24.0, 25.0, 26.0]])).all())
        self.assertTrue(F.max_pool(v, mask, -2).isclose(torch.FloatTensor([[9.0, 10.0, 11.0], [24.0, 25.0, 26.0]])).all())
        self.assertTrue(F.max_pool(v, mask, 0).isclose(torch.FloatTensor([[-1.0,-1.0,-1.0], [18.0, 19.0, 20.0], [-1.0,-1.0,-1.0], [24.0, 25.0, 26.0], [-1.0,-1.0,-1.0]])).all())
        self.assertTrue(F.max_pool(v, mask, -3).isclose(torch.FloatTensor([[-1.0,-1.0,-1.0], [18.0, 19.0, 20.0], [-1.0,-1.0,-1.0], [24.0, 25.0, 26.0], [-1.0,-1.0,-1.0]])).all())

class AveragePoolTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_average_pool(self):
        v = torch.arange(2*5*3).reshape(2,5,3).to(torch.float32)
        mask = torch.FloatTensor([[0.0, 1.0, 0.0, 1.0, 0.0]]).repeat(2,1) # [2,5]
        """
        [
            [[0,1,2],    [3,4,5],    [6,7,8],    [9,10,11],  [12,13,14]],
            [[15,16,17], [18,19,20], [21,22,23], [24,25,26], [27,28,29]],
        ]
        => [
            [[0,0,0],    [3,4,5], [0,0,0], [9,10,11],  [0,0,0]],
            [[0,0,0], [18,19,20], [0,0,0], [24,25,26], [0,0,0]],
        ]
        """
        self.assertTrue(F.average_pool(v, mask).isclose(torch.FloatTensor([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]])).all())
        self.assertTrue(F.average_pool(v, mask, 1).isclose(torch.FloatTensor([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]])).all())
        self.assertTrue(F.average_pool(v, mask, -2).isclose(torch.FloatTensor([[6.0, 7.0, 8.0], [21.0, 22.0, 23.0]])).all())
        self.assertTrue(F.average_pool(v, mask, 0).isclose(torch.FloatTensor([[0.0, 0.0, 0.0], [21.0/2, 23.0/2, 25.0/2], [0.0, 0.0, 0.0], [33.0/2, 35.0/2, 37.0/2], [0.0, 0.0, 0.0]])).all())
        self.assertTrue(F.average_pool(v, mask, -3).isclose(torch.FloatTensor([[0.0, 0.0, 0.0], [21.0/2, 23.0/2, 25.0/2], [0.0, 0.0, 0.0], [33.0/2, 35.0/2, 37.0/2], [0.0, 0.0, 0.0]])).all())




