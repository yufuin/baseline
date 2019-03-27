import numpy as np
import tensorflow as tf

from baseline.tf.utils import sinusoid_position_encoding

class BaseMultiHeadAttention(tf.keras.layers.Layer):
    """
    This is the base class for multi-head attention.
    """
    epsilon = 1e-10

    def __init__(self, attention_type:"dot-product/additive"="dot-product", use_bias=False, dim_additive=None, additive_activation="tanh", **kwargs):
        super(BaseMultiHeadAttention, self).__init__(**kwargs)

        self.attention_type = str(attention_type)
        if self.attention_type == "dot-product":
            self.attend = self.dot_product_attend
        elif self.attention_type == "additive":
            assert dim_additive is not None
            self.dim_additive = dim_additive
            self.additive_activation = tf.keras.activations.get(additive_activation)
            self.attend = self.additive_attend
        else:
            raise ValueError("invalid attention_type: " + self.attention_type)
        self.use_bias = use_bias

        self.attention_built = False

    def call(self, inputs):
        raise NotImplementedError("BaseMultiHeadAttention is the abstract class.")

    def attention_build(self, num_head=None, dim_query=None, dim_key=None):
        if self.attention_type == "dot-product":
            # if we use softmax_reduce, this bias substantially has no effect.
            if self.use_bias:
                self.attention_bias = self.add_weight(
                    name="attention_bias",
                    shape = [num_head],
                    initializer=tf.keras.initializers.zeros())

        elif self.attention_type == "additive":
            self.query_kernel = self.add_weight(
                name="query_kernel",
                shape=[num_head, dim_query, self.dim_additive],
                initializer=tf.keras.initializers.glorot_normal())
            self.key_kernel = self.add_weight(
                name="key_kernel",
                shape=[num_head, dim_key, self.dim_additive],
                initializer=tf.keras.initializers.glorot_normal())
            if self.use_bias:
                self.additive_bias = self.add_weight(
                    name="additive_bias",
                    shape = [num_head, self.dim_additive],
                    initializer=tf.keras.initializers.zeros())

            self.attention_kernel = self.add_weight(
                name="attention_kernel",
                shape=[num_head, self.dim_additive],
                initializer=tf.keras.initializers.glorot_normal())
            # if we use softmax_reduce, this bias substantially has no effect.
            if self.use_bias:
                self.attention_bias = self.add_weight(
                    name="attention_bias",
                    shape = [num_head],
                    initializer=tf.keras.initializers.zeros())
        else:
            raise ValueError("invalid attention_type: " + self.attention_type)
        self.attention_built = True

    def build(self, input_shape):
        assert self.attention_built, "must have called self.attention_build"
        super(BaseMultiHeadAttention, self).build(input_shape)

    def dot_product_attend(self, queries:"[batch_size,query_seq_len,num_head,dim_key]", keys:"[batch_size,key_seq_len,num_head,dim_key]", values:"[batch_size,key_seq_len,num_head,dim_value]", key_mask:"[batch_size,key_seq_len]"=None):
        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        dim_keys = tf.shape(keys)[-1]

        us = tf.reduce_sum(tf.expand_dims(queries, -3) * tf.expand_dims(keys, -4), axis=-1) / tf.sqrt(tf.cast(dim_keys, dtype)) # [batch_size, query_seq_len, key_seq_len, num_head]
        if self.use_bias:
            us = us + self.attention_bias

        reduction, self.attentions = self.softmax_reduce(scores=us, values=values, key_mask=key_mask)
        return reduction # [batch_size, query_seq_len, num_head*dim_value]

    def additive_attend(self, queries:"[batch_size,query_seq_len,num_head,dim_key]", keys:"[batch_size,key_seq_len,num_head,dim_key]", values:"[batch_size,key_seq_len,num_head,dim_value]", key_mask:"[batch_size,key_seq_len]"=None):
        h_query = tf.reduce_sum(tf.expand_dims(queries, -1) * self.query_kernel, axis=-2) # [batch_size, query_seq_len, num_head, dim_hidden]
        h_key = tf.reduce_sum(tf.expand_dims(keys, -1) * self.key_kernel, axis=-2) # [batch_size, key_seq_len, num_head, dim_hidden]
        h_additive = tf.expand_dims(h_query, -3) + tf.expand_dims(h_key, -4) # [batch_size, query_seq_len, key_seq_len, num_head, dim_hidden]
        if self.use_bias:
            h_additive = h_additive + self.additive_bias
        if self.additive_activation is not None:
            h_additive = self.additive_activation(h_additive)
        us = tf.reduce_sum(h_additive * self.attention_kernel, -1) # [batch_size, query_seq_len, key_seq_len, num_head]
        if self.use_bias:
            us = us + self.attention_bias

        reduction, self.attentions = self.softmax_reduce(scores=us, values=values, key_mask=key_mask)
        return reduction # [batch_size, query_seq_len, num_head*dim_value]

    def softmax_reduce(self, scores:"[batch_size, query_seq_len, key_seq_len, num_head]", values:"[batch_size, key_seq_len, num_head, dim_value]", key_mask:"[batch_size,key_seq_len]"=None):
        if key_mask is not None:
            dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
            if key_mask.dtype != dtype:
                key_mask = tf.cast(key_mask, dtype)

        scores = scores - tf.reduce_max(scores, axis=-2, keepdims=True) # to avoid overflow

        exp_scores = tf.exp(scores) # [batch_size, query_seq_len, key_seq_len, num_head]
        if key_mask is not None:
            exp_scores = exp_scores * tf.expand_dims(tf.expand_dims(key_mask, -2), -1)
        attentions = exp_scores / (tf.reduce_sum(exp_scores, axis=-2, keepdims=True) + self.epsilon) # [batch_size, query_seq_len, key_seq_len, num_head]

        reduction = tf.reduce_sum(tf.expand_dims(attentions, -1) * tf.expand_dims(values, -4), axis=-3) # [batch_size, query_seq_len, num_head, dim_value]
        assert reduction.shape[-2:].is_fully_defined()
        reduction = tf.reshape(reduction, tf.unstack(tf.shape(reduction))[:-2] + [reduction.shape[-2]*reduction.shape[-1]]) # [batch_size, query_seq_len, num_head*dim_value]
        return reduction, attentions # [batch_size, query_seq_len, num_head*dim_value], [batch_size, query_seq_len, key_seq_len, num_head]

    def get_config(self):
        config = {
            "attention_type": self.attention_type,
            "use_bias": self.use_bias
            }
        if self.attention_type == "additive":
            config["dim_additive"] = self.dim_additive
            config["additive_activation"] = tf.keras.activations.serialize(self.additive_activation)
        base_config = super(BaseMultiHeadAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadReduction(BaseMultiHeadAttention):
    def __init__(self, dim_sum_output, num_head, position_type:"none/add"="none", use_bias=False, key_activation:"callable"=None, attention_type="dot-product", **kwargs):
        assert dim_sum_output % num_head == 0
        assert position_type in ["none", "add"]

        super(MultiHeadReduction, self).__init__(attention_type=attention_type, use_bias=use_bias, **kwargs)
        self.dim_sum_output = dim_sum_output
        self.num_head = num_head
        self.dim_each_output = self.dim_sum_output // self.num_head
        self.position_type = str(position_type)
        self.key_activation = tf.keras.activations.get(key_activation)

        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        self.input_dim = input_shape[-1].value
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={2:self.input_dim})

        self.kv_kernel = self.add_weight(
            name="kv_kernel",
            shape=[self.input_dim, 2, self.num_head, self.dim_each_output],
            initializer=tf.keras.initializers.glorot_normal())
        self.query_kernel = self.add_weight(
            name="query_kernel",
            shape=[self.num_head, self.dim_each_output],
            initializer=tf.keras.initializers.random_normal(mean=0.0, stddev=1.0))
        if self.use_bias:
            self.value_bias = self.add_weight(
                name="value_bias",
                shape = [self.dim_sum_output],
                initializer=tf.keras.initializers.zeros())

        super(MultiHeadReduction, self).attention_build(num_head=self.num_head, dim_query=self.dim_each_output, dim_key=self.dim_each_output)
        super(MultiHeadReduction, self).build(input_shape)

    def call(self, inputs:"[batch_size,seq_len,input_dim]", mask:"[batch_size, seq_len]"=None):
        if self.position_type == "add":
            max_seq_len = tf.shape(inputs)[1]
            dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
            input_dim = inputs.shape[2] if inputs.shape[2].value is not None else tf.shape(inputs)[2]
            inputs = inputs + sinusoid_position_encoding(sequence_length=max_seq_len, dim=input_dim, dtype=dtype)

        kvs = tf.tensordot(inputs, self.kv_kernel, 1) # [batch_size, seq_len, 2, num_head, dim_each_output]
        keys, values = tf.unstack(kvs, axis=2) # 2x[batch_size, seq_len, num_head, dim_each_output]
        if self.key_activation is not None:
            keys = self.key_activation(keys)

        reduction = tf.squeeze(self.dot_product_attend(keys=keys, values=values, queries=self.query_kernel, key_mask=mask), 1) # [batch_size, dim_sum_output]
        if self.use_bias:
            reduction = reduction + self.value_bias
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
            "key_activation": tf.keras.activations.serialize(self.key_activation)
            }
        base_config = super(MultiHeadReduction, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class MultiHeadSelfAttention(BaseMultiHeadAttention):
    def __init__(self, dim_sum_output, num_head, position_type:"none/add"="none", use_bias=False, attention_type="dot-product", **kwargs):
        assert dim_sum_output % num_head == 0
        assert position_type in ["none", "add"]

        super(MultiHeadSelfAttention, self).__init__(attention_type=attention_type, use_bias=use_bias, **kwargs)
        self.dim_sum_output = dim_sum_output
        self.num_head = num_head
        self.dim_each_output = self.dim_sum_output // self.num_head
        self.position_type = str(position_type)

        self.supports_masking = True
        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        self.input_dim = input_shape[-1].value
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={2:self.input_dim})

        self.kvq_kernel = self.add_weight(
            name="kvq_kernel",
            shape=[self.input_dim, 3, self.num_head, self.dim_each_output],
            initializer=tf.keras.initializers.glorot_normal())
        if self.use_bias:
            self.value_bias = self.add_weight(
                name="value_bias",
                shape = [self.dim_sum_output],
                initializer=tf.keras.initializers.zeros())

        super(MultiHeadSelfAttention, self).attention_build(num_head=self.num_head, dim_query=self.dim_each_output, dim_key=self.dim_each_output)
        super(MultiHeadSelfAttention, self).build(input_shape)

    def call(self, inputs:"[batch_size,seq_len,input_dim]", mask:"[batch_size, seq_len]"=None):
        if self.position_type == "add":
            max_seq_len = tf.shape(inputs)[1]
            dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
            input_dim = inputs.shape[2] if inputs.shape[2].value is not None else tf.shape(inputs)[2]
            inputs = inputs + sinusoid_position_encoding(sequence_length=max_seq_len, dim=input_dim, dtype=dtype)

        kvqs = tf.tensordot(inputs, self.kvq_kernel, 1) # [batch_size, seq_len, 3, num_head, dim_each_output]
        keys, values, queries = tf.unstack(kvqs, axis=2) # 3x[batch_size, seq_len, num_head, dim_each_output]

        reduction = self.dot_product_attend(keys=keys, values=values, queries=queries, key_mask=mask) # [batch_size, seq_len, dim_sum_output]
        if self.use_bias:
            reduction = reduction + self.value_bias
        return reduction # [batch_size, dim_sum_output]

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        return input_shape[:2].concatenate(self.dim_sum_output)

    #def compute_mask(self, inputs, mask): same mask, not need to override.

    def get_config(self):
        config = {
            "dim_sum_output": self.dim_sum_output,
            "num_head": self.num_head,
            "position_type": self.position_type,
            }
        base_config = super(MultiHeadSelfAttention, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))





