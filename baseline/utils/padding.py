from itertools import zip_longest

def get_padded_shape(seq):
    assert type(seq) in [list, tuple]
    shape = _get_padded_shape(seq)
    return shape
def _get_padded_shape(seq, _current_depth=0):
    if any(type(deep) in [list, tuple] for deep in seq):
        deep_shapes = [_get_padded_shape(deep_seq, _current_depth=_current_depth+1) for deep_seq in seq]
        deep_shape = [max(lens) for lens in zip_longest(*deep_shapes, fillvalue=0)]
        return [len(seq), *deep_shape]
    else:
        return [len(seq)]

def pad(seq, padding_value):
    assert type(seq) is list
    shape = get_padded_shape(seq)
    padded, mask = _pad(seq, padding_value=padding_value, shape=shape)
    return padded, mask
def _pad(seq, padding_value, shape, _current_depth=0):
    assert type(seq) is list
    if _current_depth == len(shape) - 1:
        pad_vec = [padding_value]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = [1] * len(seq) + [0] * len(pad_vec)
        return seq + pad_vec, mask
    else:
        deep_seqs_and_masks = [_pad(deep_seq, padding_value=padding_value, shape=shape, _current_depth=_current_depth+1) for deep_seq in seq]
        if len(deep_seqs_and_masks) > 0:
            deep_seqs, deep_mask = map(list, list(zip(*deep_seqs_and_masks)))
        else:
            deep_seqs, deep_mask = [], []

        pad_vec = [padding_value]
        zero_vec = [0]
        for l in reversed(shape[_current_depth+1:]):
            pad_vec = pad_vec * l
            zero_vec = zero_vec * l
            pad_vec = [pad_vec]
            zero_vec = [zero_vec]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = deep_mask + zero_vec * (shape[_current_depth] - len(seq))
        return deep_seqs + pad_vec, mask
