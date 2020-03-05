from torch.nn.functional import elu

def elu_clip(arg, lower=-15.0, upper=15.0):
    """
    lower-1 <= cliped <= upper+1
    """
    return upper - elu((upper - lower) - elu(arg-lower))

def flatten(target, indices):
    """
    v = torch.arange(2*3*5*7).reshape(2,3,5,7)
    v.shape # [2,3,5,7]
    flatten(v, [0,1]) # [2x3, 5, 7]
    flatten(v, [1,2]) # [2, 3x5, 7]
    flatten(v, [1,2,3]) # [2, 3x5x7]
    """
    assert type(indices) in [list, tuple] and len(indices) >= 2
    indices = sorted(indices)
    assert all(indices[i+1]-indices[i] == 1 for i in range(len(indices)-1))
    min_index, max_index = indices[0], indices[-1]
    in_shape = list(target.shape)
    out_shape = in_shape[:min_index] + [-1] + in_shape[max_index+1:]
    return target.reshape(out_shape)
