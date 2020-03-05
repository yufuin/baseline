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


