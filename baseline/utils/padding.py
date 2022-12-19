from itertools import zip_longest as _zip_longest
import numpy as _np

def get_padded_shape(seq):
    assert type(seq) in [list, tuple]
    shape, depth, is_empty = _get_padded_shape(seq)
    assert len(shape) == depth
    return shape
def _get_padded_shape(seq, current_depth=1):
    if any(type(deep) in [list, tuple] for deep in seq):
        deep_shapes, depths, is_empties = zip(*[_get_padded_shape(deep_seq, current_depth=current_depth+1) for deep_seq in seq])
        max_depth = max(depths)
        assert all(max_depth == depth for depth,is_empty in zip(depths, is_empties) if not is_empty), (depths, is_empties, seq)
        deep_shape = [max(lens) for lens in _zip_longest(*deep_shapes, fillvalue=0)]
        is_all_empty = all(is_empties)
        return [len(seq), *deep_shape], max_depth, is_all_empty
    else:
        is_empty = len(seq) == 0
        return [len(seq)], current_depth, is_empty

def pad(seq, padding_value):
    assert type(seq) is list
    shape = get_padded_shape(seq)
    padded, mask = _pad(seq, padding_value=padding_value, shape=shape)
    return padded, mask
def _pad(seq, padding_value, shape, current_depth=0):
    assert type(seq) is list
    if current_depth == len(shape) - 1:
        pad_vec = [padding_value]
        pad_vec = pad_vec * (shape[current_depth] - len(seq))
        mask = [1] * len(seq) + [0] * len(pad_vec)
        return seq + pad_vec, mask
    else:
        deep_seqs_and_masks = [_pad(deep_seq, padding_value=padding_value, shape=shape, current_depth=current_depth+1) for deep_seq in seq]
        if len(deep_seqs_and_masks) > 0:
            deep_seqs, deep_mask = map(list, list(zip(*deep_seqs_and_masks)))
        else:
            deep_seqs, deep_mask = [], []

        pad_vec = [padding_value]
        zero_vec = [0]
        for l in reversed(shape[current_depth+1:]):
            pad_vec = pad_vec * l
            zero_vec = zero_vec * l
            pad_vec = [pad_vec]
            zero_vec = [zero_vec]
        pad_vec = pad_vec * (shape[current_depth] - len(seq))
        mask = deep_mask + zero_vec * (shape[current_depth] - len(seq))
        return deep_seqs + pad_vec, mask



def get_padded_shape_numpy(seq):
    assert type(seq) in [list, tuple]
    shape, depth, is_empty = _get_padded_shape_numpy(seq)
    assert len(shape) == depth
    return shape
def _get_padded_shape_numpy(seq, current_depth=1):
    if type(seq) in [list, tuple]:
        is_empty = len(seq) == 0
        if is_empty:
            return [0], current_depth, is_empty
        else:
            deep_shapes, depths, is_empties = zip(*[_get_padded_shape_numpy(deep_seq, current_depth=current_depth+1) for deep_seq in seq])
            max_depth = max(depths)
            assert all(max_depth == depth for depth,is_empty in zip(depths, is_empties) if not is_empty), (depths, is_empties, seq)
            deep_shape = [max(lens) for lens in _zip_longest(*deep_shapes, fillvalue=0)]
            is_all_empty = all(is_empties)
            return [len(seq), *deep_shape], max_depth, is_all_empty
    else:
        assert type(seq) is _np.ndarray, seq
        shape = list(seq.shape)
        depth = len(shape) + current_depth - 1
        is_empty = False
        return shape, depth, is_empty

def _check_dtype(seq):
    if type(seq) in [list, tuple]:
        dtypes = [_check_dtype(v) for v in seq]
        dtypes = [dtype for dtype in dtypes if dtype is not None]
        if len(dtypes) == 0:
            return None
        else:
            first_dtype = dtypes[0]
            assert all(dtype == first_dtype for dtype in dtypes), dtypes
            return first_dtype
    else:
        assert type(seq) is _np.ndarray
        return seq.dtype

def pad_numpy(seq, padding_value):
    assert type(seq) in [list, tuple]
    shape = get_padded_shape_numpy(seq)
    dtype = _check_dtype(seq)
    if dtype is None:
        if type(padding_value) is float:
            dtype = _np.float32
        else:
            dtype = _np.int64
    padded, mask = _pad_numpy(seq, padding_value=padding_value, shape=shape, dtype=dtype)
    padded = _np.stack(padded)
    mask = _np.stack(mask)
    return padded, mask
def _pad_numpy(seq, padding_value, shape, dtype, current_depth=0):
    if type(seq) in [list, tuple]:
        if len(seq) == 0:
            padded = _np.full(shape[current_depth:], padding_value, dtype=dtype)
            mask = _np.zeros(shape[current_depth:], dtype=dtype)
            return padded, mask
        else:
            padded, mask = zip(*[_pad_numpy(v, padding_value=padding_value, shape=shape, dtype=dtype, current_depth=current_depth+1) for v in seq])
            padded, mask = map(_np.stack, [padded, mask])
            num_missing = shape[current_depth] - len(padded)
            assert list(padded.shape[1:]) == shape[current_depth+1:]
            if num_missing > 0:
                padded = _np.pad(padded, [[0, num_missing]] + [[0,0] for _ in range(len(shape)-current_depth-1)], mode="constant", constant_values=padding_value)
                mask = _np.pad(mask, [[0, num_missing]] + [[0,0] for _ in range(len(shape)-current_depth-1)], mode="constant", constant_values=0)
            return padded, mask
    else:
        assert type(seq) is _np.ndarray
        assert len(seq.shape) == (len(shape)-current_depth)
        padded = _np.pad(seq, [[0,dest_size-src_size] for src_size, dest_size in zip(seq.shape, shape[current_depth:])], mode="constant", constant_values=padding_value)
        mask = _np.pad(_np.ones_like(seq), [[0,dest_size-src_size] for src_size, dest_size in zip(seq.shape, shape[current_depth:])], mode="constant", constant_values=0)
        return padded, mask
