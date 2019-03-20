import unittest

import tensorflow as tf
tf.enable_eager_execution()
def to_raw(tensor):
    return tensor.numpy().tolist()

from baseline.tensorflow import utils as U

class EluClipTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_elu_clip(self):
        # -16.0 <= v <= 16.0 => elu_clip(v) == v
        self.assertAlmostEqual(0.0, to_raw(U.elu_clip(0.0)))
        for v in [0.0, 1.0, 8.1, 14.4, 16.0]:
            self.assertAlmostEqual(v, to_raw(U.elu_clip(v, -16.0, 16.0)), delta=v*1e-5)
            self.assertAlmostEqual(-v, U.elu_clip(-v, -16.0, 16.0), delta=v*1e-5)

        # v < -16.0 => -17.0 < elu_clip(v, -16.0, 16.0) < -16.0
        # 16.0 < v => 16.0 < elu_clip(v, -16.0, 16.0) < 17.0
        for v in [16.5, 17.0, 20.0]:
            self.assertNotAlmostEqual(v, to_raw(U.elu_clip(v, -16.0, 16.0)), delta=v*1e-5)
            self.assertLess(16.0, to_raw(U.elu_clip(v, -16.0, 16.0)))
            self.assertLess(to_raw(U.elu_clip(v, -16.0, 16.0)), 17.0)

            self.assertNotAlmostEqual(-v, U.elu_clip(-v, -16.0, 16.0), delta=v*1e-5)
            self.assertLess(-17.0, to_raw(U.elu_clip(-v, -16.0, 16.0)))
            self.assertLess(to_raw(U.elu_clip(-v, -16.0, 16.0)), -16.0)

