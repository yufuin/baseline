import unittest

from baseline.utils.argparse import DataclassArgumentParser
import dataclasses

class ClosedIntervalTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_dataclass_argumentparser1(self):
        @dataclasses.dataclass
        class Data:
            x:int = 3
            y:float = dataclasses.field(default=1e-2, metadata={"choices":[1e-2, 1e-4, 1e-6]})
            z:str = dataclasses.field(default_factory=lambda:None, metadata={"args":["--z", "--str"]})
        parser = DataclassArgumentParser(Data)

        parsed = parser.parse_args(["--x", "7"])
        must_same = Data(x=7, y=1e-2, z=None)
        self.assertEqual(parsed, must_same)

        parsed = parser.parse_args(["--y", "1e-4", "--z", "foobar"])
        must_same = Data(y=1e-4, z="foobar")
        self.assertEqual(parsed, must_same)

