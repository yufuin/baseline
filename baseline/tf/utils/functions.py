import tensorflow as tf

def elu_clip(arg, lower=-15.0, upper=15.0):
    """
    lower-1 <= cliped <= upper+1
    """
    return upper - tf.nn.elu((upper - lower) - tf.nn.elu(arg-lower))

def sinusoid_position_encoding(sequence_length:"[]", dim:"[]", dtype=tf.float32, wave=10000) -> "[sequence_length, dim]":
    half_dim = dim // 2
    d = tf.cast(tf.range(half_dim), dtype) # [dim//2]
    div = tf.pow(tf.cast(wave, dtype), d / tf.cast(half_dim, dtype=dtype)) # [dim//2]
    position = tf.cast(tf.range(sequence_length), dtype) # [seq_len]
    theta = tf.expand_dims(position, 1) / tf.expand_dims(div, 0) # [seq_len, dim//2]
    sin = tf.sin(theta)
    cos = tf.cos(theta)
    encoded = tf.concat([sin, cos], axis=1) # [seq_len, dim]
    encoded = tf.identity(encoded, name="sinusoid_position_encoding")
    return encoded


