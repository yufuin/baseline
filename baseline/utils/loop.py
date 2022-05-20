class _Import:
    import numpy as np
    import typing
    from typing import Optional, Union, Any

class Loop:
    def __init__(self, values, shuffle:bool, *_, post_mapping:_Import.Optional[_Import.Any]=None, random_state:_Import.Optional[_Import.Union[int, _Import.np.random.RandomState]]=None):
        assert len(values) > 0
        assert len(_) == 0, f"invalid argument: {_}"
        self.values = values
        self._len = len(self.values)
        self.shuffle = shuffle
        self.post_mapping = post_mapping
        self.random_state = random_state

        self.step = 0
        self.orders = list(range(self._len))
        if self.shuffle:
            if self.random_state is None:
                self.rng = _Import.np.random.RandomState()
            elif isinstance(self.random_state, int):
                self.rng = _Import.np.random.RandomState(self.random_state)
            else:
                self.rng = self.random_state
        self.reset()

    def reset(self):
        self.step = 0
        if self.shuffle:
            self.rng.shuffle(self.orders)

    def get(self, size):
        assert size >= 0
        if self.step+size > self._len:
            f_size = self._len - self.step
            b_size = size - f_size
            f_out = self.get(f_size)
            self.reset()
            return f_out + self.get(b_size)
        else:
            out = [self.values[self.orders[i]] for i in range(self.step, self.step+size)]
            if self.post_mapping is not None:
                out = [self.post_mapping(value) for value in out]
            self.step += size
            return out
