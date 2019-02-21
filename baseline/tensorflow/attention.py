import numpy as np
import tensorflow as tf

def sinusoid_position_encoding(sequence_length:"[]", dim:"[]", dtype=tf.float32, wave=10000) -> "[sequence_length, dim]":
    half_dim = tf.cast(dim // 2, dtype)
    position = tf.range(tf.cast(sequence_length, dtype)) # [seq_len]
    d = tf.range(half_dim) # [dim//2]
    div = tf.pow(tf.cast(wave, dtype), d / half_dim) # [dim//2]
    theta = position[:,tf.newaxis] / div[tf.newaxis,:] # [seq_len, dim//2]
    sin = tf.sin(theta)
    cos = tf.cos(theta)
    encoded = tf.concat([sin, cos], axis=1) # [seq_len, dim]
    encoded = tf.identity(encoded, name="sinusoid_position_encoding")
    return encoded

class MultiHeadReduction:
    clip_max = 20.0
    epsilon = 1e-10
    def __init__(self, dim_sum_output, num_head, position_type:"none/add"="none", use_bias=False, key_activation:"callable"=None):
        assert dim_sum_output % num_head == 0
        assert position_type in ["none", "add"]

        self.dim_sum_output = dim_sum_output
        self.num_head = num_head
        self.dim_each_output = self.dim_sum_output // self.num_head
        self.position_type = str(position_type)
        self.use_bias = use_bias
        self.key_activation = key_activation

        initializer = tf.glorot_normal_initializer()
        self.fc_kvs = tf.keras.layers.Dense(2*self.dim_sum_output, use_bias=self.use_bias, kernel_initializer=initializer)
        self.w_queries = tf.Variable(tf.random_normal_initializer(mean=0.0, stddev=1.0)([self.num_head, self.dim_each_output]))
        if self.use_bias:
            self.b_queries = tf.Variable(tf.zeros([self.num_head, 1]))
        self._root_d_k = tf.constant(np.sqrt(self.dim_each_output), dtype=tf.float32)

    def __call__(self, inputs:"[batch_size,seq_len,dim_input]", seq_lens:"[batch_size]", batch_size:"[]"=None, max_seq_len:"[]"=None):
        if batch_size is None: batch_size = tf.shape(inputs)[0]
        if max_seq_len is None: max_seq_len = tf.shape(inputs)[1]
        seq_mask = tf.sequence_mask(seq_lens, max_seq_len, dtype=tf.float32) # [batch_size, seq_len]

        if self.position_type == "add":
            dim_input = inputs.shape[2] if inputs.shape[2].value is not None else tf.shape(inputs)[2]
            inputs = inputs + sinusoid_position_encoding(sequence_length=max_seq_len, dim=dim_input)

        concat_kvs = self.fc_kvs(inputs) # [batch_size, seq_len, 2*num_head*dim_each_output]
        kvs = tf.reshape(concat_kvs, [batch_size, max_seq_len, 2*self.num_head, self.dim_each_output])
        keys, values = tf.split(kvs, 2, axis=2) # 2x[batch_size, seq_len, num_head, dim_each_output]
        if self.key_activation is not None:
            keys = self.key_activation(keys)

        us = tf.reduce_sum(keys*self.w_queries, axis=-1, keepdims=True) # [batch_size, seq_len, num_head, 1]
        if self.use_bias:
            us = us + self.b_queries
        us = tf.clip_by_value(us / self._root_d_k, -self.clip_max, self.clip_max) # [batch_size, seq_len, num_head, 1]
        exp_us = tf.exp(us) * seq_mask[:,:,tf.newaxis,tf.newaxis] # [batch_size, seq_len, num_head, 1]
        attentions = exp_us / (tf.reduce_sum(exp_us, axis=1, keepdims=True) + self.epsilon) # [batch_size, seq_len, num_head, 1]

        reduction = tf.reshape(tf.reduce_sum(values * attentions, axis=1), [batch_size, self.dim_sum_output]) # [batch_size, dim_sum_output]
        self.attentions = tf.squeeze(attentions, axis=-1) # [batch_size, seq_len, num_head]
        return reduction # [batch_size, dim_sum_output]

class MultiHeadSelfAttention:
    clip_max = 20.0
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

        us = tf.clip_by_value(tf.reduce_sum(keys[:,tf.newaxis]*queries[:,:,tf.newaxis], axis=-1, keepdims=True) / self._root_d_k, -self.clip_max, self.clip_max) # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        exp_us = tf.exp(us) * seq_mask[:,:,tf.newaxis,tf.newaxis,tf.newaxis] * seq_mask[:,tf.newaxis,:,tf.newaxis,tf.newaxis] # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        attentions = exp_us / (tf.reduce_sum(exp_us, axis=2, keepdims=True) + self.epsilon) # [batch_size, query_seq_len, key_seq_len, num_head, 1]

        reduction = tf.reshape(tf.reduce_sum(values[:,tf.newaxis] * attentions, axis=2), [batch_size, max_seq_len, self.dim_sum_output])
        self.attentions = tf.squeeze(attentions, axis=-1) # [batch_size, query_seq_len, key_seq_len, num_head]
        return reduction # [batch_size, max_seq_len, dim_sum_output]

