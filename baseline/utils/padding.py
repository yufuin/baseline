def get_padded_shape(seq):
    assert type(seq) in [list, tuple]
    shape = _get_padded_shape(seq)
    return shape
def _get_padded_shape(seq, _current_depth=0):
    if type(seq[0]) in [list, tuple]:
        deep_shapes = [_get_padded_shape(deep_seq, _current_depth=_current_depth+1) for deep_seq in seq]
        deep_shape = [max(lens) for lens in zip(*deep_shapes)]
        return [len(seq), *deep_shape]
    else:
        return [len(seq)]

def pad(seq, padded_value):
    assert type(seq) is list
    shape = get_padded_shape(seq)
    padded, mask = _pad(seq, padded_value=padded_value, shape=shape)
    return padded, mask
def _pad(seq, padded_value, shape, _current_depth=0):
    assert type(seq) is list
    if _current_depth == len(shape) - 1:
        pad_vec = [padded_value]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = [1] * len(seq) + [0] * len(pad_vec)
        return seq + pad_vec, mask
    else:
        deep_seqs_and_masks = [_pad(deep_seq, padded_value=padded_value, shape=shape, _current_depth=_current_depth+1) for deep_seq in seq]
        deep_seqs, deep_mask = map(list, list(zip(*deep_seqs_and_masks)))

        pad_vec = [padded_value]
        zero_vec = [0]
        for l in reversed(shape[_current_depth+1:]):
            pad_vec = pad_vec * l
            zero_vec = zero_vec * l
            pad_vec = [pad_vec]
            zero_vec = [zero_vec]
        pad_vec = pad_vec * (shape[_current_depth] - len(seq))
        mask = deep_mask + zero_vec * (shape[_current_depth] - len(seq))
        return deep_seqs + pad_vec, mask
