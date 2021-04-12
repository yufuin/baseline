import tqdm
import contextlib

def closing_tqdm(*args, **kwargs):
    return contextlib.closing(tqdm.tqdm(*args, **kwargs))

def closing_tqdm_iter(*args, **kwargs):
    with closing_tqdm(*args, **kwargs) as it:
        for i in it:
            yield i
