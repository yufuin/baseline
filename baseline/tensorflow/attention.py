import numpy as np
import tensorflow as tf

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


class MultiHeadReduction(tf.keras.layers.Layer):
    epsilon = 1e-10
    def __init__(self, dim_sum_output, num_head, position_type:"none/add"="none", use_bias=False, key_activation:"callable"=None, **kwargs):
        assert dim_sum_output % num_head == 0
        assert position_type in ["none", "add"]

        super(MultiHeadReduction, self).__init__(**kwargs)
        self.dim_sum_output = dim_sum_output
        self.num_head = num_head
        self.dim_each_output = self.dim_sum_output // self.num_head
        self.position_type = str(position_type)
        self.use_bias = use_bias
        self.key_activation = tf.keras.activations.get(key_activation)

        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        self.dim_input = input_shape[-1].value
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={2:self.dim_input})

        self.kv_kernel = self.add_weight(
            name="kv_kernel",
            shape=[self.dim_input, 2*self.dim_sum_output],
            initializer=tf.keras.initializers.glorot_normal())
        if self.use_bias:
            self.kv_bias = self.add_weight(
                name="kv_bias",
                shape = [2*self.dim_sum_output],
                initializer=tf.keras.initializers.zeros())

        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_head, self.dim_each_output],
            initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))
        if self.use_bias:
            self.query_bias = self.add_weight(
                name="query_bias",
                shape = [self.num_head],
                initializer=tf.keras.initializers.zeros())

        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        self._root_d_k = tf.constant(np.sqrt(self.dim_each_output), dtype=dtype)

        super(MultiHeadReduction, self).build(input_shape)

    def call(self, inputs:"[batch_size,seq_len,dim_input]", mask:"[batch_size, seq_len]"=None):
        shape = tf.shape(inputs)
        batch_size = shape[0]
        max_seq_len = shape[1]
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())

        if mask is not None:
            if mask.dtype != dtype:
                mask = tf.cast(mask, dtype)
            seq_mask = tf.expand_dims(mask, 2)

        if self.position_type == "add":
            dim_input = inputs.shape[2] if inputs.shape[2].value is not None else tf.shape(inputs)[2]
            inputs = inputs + sinusoid_position_encoding(sequence_length=max_seq_len, dim=dim_input, dtype=dtype)

        concat_kvs = tf.tensordot(inputs, self.kv_kernel, 1) # [batch_size, seq_len, 2*num_head*dim_each_output]
        if self.use_bias:
            concat_kvs = concat_kvs + self.kv_bias
        kvs = tf.reshape(concat_kvs, [batch_size, max_seq_len, 2*self.num_head, self.dim_each_output])
        keys, values = tf.split(kvs, 2, axis=2) # 2x[batch_size, seq_len, num_head, dim_each_output]
        if self.key_activation is not None:
            keys = self.key_activation(keys)

        us = tf.reduce_sum(keys*self.query_kernel, axis=-1) # [batch_size, seq_len, num_head]
        if self.use_bias:
            us = us + self.query_bias
        us = us / self._root_d_k # [batch_size, seq_len, num_head]
        us = us - tf.reduce_max(us, axis=1, keepdims=True) # to avoid overflow

        exp_us = tf.exp(us) # [batch_size, seq_len, num_head]
        if mask is not None:
            exp_us = exp_us * seq_mask
        self.attentions = exp_us / (tf.reduce_sum(exp_us, axis=1, keepdims=True) + self.epsilon) # [batch_size, seq_len, num_head]

        reduction = tf.reshape(tf.reduce_sum(values * tf.expand_dims(self.attentions, 3), axis=1), [batch_size, self.dim_sum_output]) # [batch_size, dim_sum_output]
        return reduction # [batch_size, dim_sum_output]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        return input_shape[:1].concatenate(self.dim_sum_output)

    def compute_mask(self, inputs, mask=None):
        return # no more mask

    def get_config(self):
        config = {
            "dim_sum_output": self.dim_sum_output,
            "num_head": self.num_head,
            "position_type": self.position_type,
            "use_bias": self.use_bias,
            "key_activation": tf.keras.activations.serialize(self.key_activation)
            }
        base_config = super(MultiHeadReduction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadSelfAttention:
    epsilon = 1e-10
    def __init__(self, dim_sum_output, num_head, position_type:"none/add"="none", use_bias=False):
        assert dim_sum_output % num_head == 0
        assert position_type in ["none", "add"]

        self.dim_sum_output = dim_sum_output
        self.num_head = num_head
        self.dim_each_output = self.dim_sum_output // self.num_head
        self.position_type = str(position_type)
        self.use_bias = use_bias

        initializer = tf.glorot_normal_initializer()
        self.fc_kvqs = tf.keras.layers.Dense(3*self.dim_sum_output, use_bias=self.use_bias, kernel_initializer=initializer)
        self._root_d_k = tf.constant(np.sqrt(self.dim_each_output), dtype=tf.float32)

    def __call__(self, inputs:"[batch_size,seq_len,dim_input]", seq_lens:"[batch_size]", batch_size:"[]"=None, max_seq_len:"[]"=None):
        if batch_size is None: batch_size = tf.shape(inputs)[0]
        if max_seq_len is None: max_seq_len = tf.shape(inputs)[1]
        seq_mask = tf.sequence_mask(seq_lens, max_seq_len, dtype=tf.float32) # [batch_size, seq_len]

        if self.position_type == "add":
            dim_input = inputs.shape[2] if inputs.shape[2].value is not None else tf.shape(inputs)[2]
            inputs = inputs + sinusoid_position_encoding(sequence_length=max_seq_len, dim=dim_input)

        concat_kvqs = self.fc_kvqs(inputs) # [batch_size, seq_len, 3*num_head*dim_each_output]
        kvqs = tf.reshape(concat_kvqs, [batch_size, max_seq_len, 3*self.num_head, self.dim_each_output])
        keys, values, queries = tf.split(kvqs, 3, axis=2) # 3x[batch_size, seq_len, num_head, dim_each_output]

        us = tf.reduce_sum(keys[:,tf.newaxis]*queries[:,:,tf.newaxis], axis=-1, keepdims=True) / self._root_d_k # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        us = us - tf.reduce_max(us, axis=2, keepdims=True) # to avoid overflow
        exp_us = tf.exp(us) * seq_mask[:,:,tf.newaxis,tf.newaxis,tf.newaxis] * seq_mask[:,tf.newaxis,:,tf.newaxis,tf.newaxis] # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        attentions = exp_us / (tf.reduce_sum(exp_us, axis=2, keepdims=True) + self.epsilon) # [batch_size, query_seq_len, key_seq_len, num_head, 1]

        reduction = tf.reshape(tf.reduce_sum(values[:,tf.newaxis] * attentions, axis=2), [batch_size, max_seq_len, self.dim_sum_output])
        self.attentions = tf.squeeze(attentions, axis=-1) # [batch_size, query_seq_len, key_seq_len, num_head]
        return reduction # [batch_size, max_seq_len, dim_sum_output]





