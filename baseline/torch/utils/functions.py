from torch.nn.functional import elu

def elu_clip(arg, lower=-15.0, upper=15.0):
    """
    lower-1 <= cliped <= upper+1
    """
    return upper - elu((upper - lower) - elu(arg-lower))

