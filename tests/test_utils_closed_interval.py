import unittest

from baseline.utils.closed_interval import ClosedInterval

class ClosedIntervalTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_contains(self):
        i1 = ClosedInterval(-3, 10)

        self.assertTrue(i1 in i1)

        self.assertTrue(1 in i1)
        self.assertTrue(-3 in i1)
        self.assertTrue(10 in i1)
        self.assertFalse(-4 in i1)
        self.assertFalse(-3.1 in i1)
        self.assertFalse(10.1 in i1)
        self.assertFalse(11 in i1)

        self.assertTrue(ClosedInterval(5,7) in i1)
        self.assertTrue(ClosedInterval(-3,7) in i1)
        self.assertTrue(ClosedInterval(5,10) in i1)
        self.assertFalse(ClosedInterval(-8,13) in i1)
        self.assertFalse(ClosedInterval(-3.1,7) in i1)
        self.assertFalse(ClosedInterval(5,11) in i1)
        self.assertFalse(ClosedInterval(5,10.1) in i1)

    def test_eq(self):
        i1 = ClosedInterval(-3, 10)

        self.assertTrue(i1 == i1)

        self.assertTrue(5 == i1)
        self.assertTrue(-3 == i1)
        self.assertTrue(10 == i1)
        self.assertTrue(-4 != i1)
        self.assertTrue(-3.1 != i1)
        self.assertTrue(11 != i1)
        self.assertTrue(10.1 != i1)

        self.assertTrue(i1 == ClosedInterval(-3, 10))
        self.assertTrue(i1 != ClosedInterval(5, 7))
        self.assertTrue(i1 != ClosedInterval(-4, 10))
        self.assertTrue(i1 != ClosedInterval(5, 10))
        self.assertTrue(i1 != ClosedInterval(-3, 11))
        self.assertTrue(i1 != ClosedInterval(-3, 7))




