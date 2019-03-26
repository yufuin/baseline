import tensorflow as tf


class BottomUpChildSumTreeLSTM(tf.keras.layers.Layer):
    def __init__(self, units, **kwargs):
        super(BottomUpChildSumTreeLSTM, self).__init__(**kwargs)

        self.units = units

        self.input_spec = tf.keras.layers.InputSpec(ndim=3)

    def build(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        self.input_dim = input_shape[-1].value
        self.input_spec = tf.keras.layers.InputSpec(ndim=3, axes={2:self.input_dim})

        self.x_fiou_kernel = self.add_weight(
            name="x_fiou_kernel",
            shape=[self.input_dim, 4*self.units],
            initializer=tf.keras.initializers.glorot_normal())
        self.h_f_kernel = self.add_weight(
            name="h_f_kernel",
            shape=[self.units, self.units],
            initializer=tf.keras.initializers.glorot_normal())
        self.h_iou_kernel = self.add_weight(
            name="h_iou_kernel",
            shape=[self.units, 3*self.units],
            initializer=tf.keras.initializers.glorot_normal())
        self.fiou_bias = self.add_weight(
            name="fiou_bias",
            shape=[4*self.units],
            initializer=tf.keras.initializers.zeros())

        super(BottomUpChildSumTreeLSTM, self).build(input_shape)

    def call(self, inputs, parents, post_orders):
        """
        If parents[b,n] == -1, it indicates the corresponding node has no parent. (i.e., root node)
        If post_orders[b,o] == -1, it is considered as the padded value
                                   and the correspondig calculation will be skipped.
        """
        output_shape = tf.unstack(tf.shape(inputs))[:-1] + [self.units]
        batch_size, max_order_len = tf.unstack(tf.shape(post_orders))
        range_batch_size = tf.range(batch_size) # [batch_size]
        def batch_indicator(indices):
            return tf.stack([range_batch_size, indices], axis=1)
        tiled_range_batch_size = tf.tile(tf.expand_dims(range_batch_size, 1), [1, max_order_len]) # [batch_size, order_len]

        dtype = tf.dtypes.as_dtype(self.dtype or tf.keras.backend.floatx())
        order_mask = tf.greater_equal(post_orders, 0)
        need_to_proceed = tf.cast(tf.reduce_any(order_mask, axis=0), dtype=max_order_len.dtype) # [order_len]
        max_step = tf.cast(tf.argmax(need_to_proceed * tf.range(max_order_len)), dtype=max_order_len.dtype) + 1
        sorted_output_mask = tf.cast(order_mask, dtype) # [batch_size, order_len]

        post_orders = tf.maximum(post_orders, 0)
        post_order_sorter = tf.stack([tiled_range_batch_size, post_orders], axis=2) # [batch_size, order_len, 2]

        sorted_parents = tf.gather_nd(params=parents, indices=post_order_sorter) # [batch_size, order_len]
        sorted_propagete_parent_mask = tf.greater_equal(sorted_parents, 0) # i.e., has_parent
        sorted_propagete_parent_mask = tf.cast(tf.logical_and(sorted_propagete_parent_mask, order_mask), dtype) # [batch_size, order_len]
        sorted_parents = tf.maximum(sorted_parents, 0)

        fiou_xb = tf.tensordot(inputs, self.x_fiou_kernel, 1) + self.fiou_bias # [batch_size, seq_len, 4*units]
        f_xb, iou_xb = tf.split(fiou_xb, [self.units, 3*self.units], axis=-1) # [batch_size, seq_len, units], [batch_size, seq_len, 3*units]

        def aggregate_nodes(target_iou_xb, target_child_sum_hs, target_gated_child_sum_cs):
            iou_h = tf.matmul(target_child_sum_hs, self.h_iou_kernel)
            iou = target_iou_xb + iou_h
            i, o, u = tf.split(iou, 3, axis=1)
            memory = tf.nn.sigmoid(i) * tf.nn.tanh(u) + target_gated_child_sum_cs
            output = tf.nn.sigmoid(o) * tf.nn.tanh(memory)
            return output, memory
        def gate_memory(output, memory, parent_f_xb):
            parent_f_h = tf.matmul(output, self.h_f_kernel)
            parent_f = parent_f_h + parent_f_xb
            gated_memory = tf.nn.sigmoid(parent_f) * memory
            return gated_memory

        sorted_output_mask = tf.expand_dims(sorted_output_mask, 2) # [batch_size, order_len, 1]
        sorted_propagete_parent_mask = tf.expand_dims(sorted_propagete_parent_mask, 2) # [batch_size, order_len, 1]
        def loop(step, hs, child_sum_hs, gated_child_sum_cs):
            step_post_orders = post_orders[:, step] # TODO: transpose
            step_target_indicators = batch_indicator(step_post_orders)
            step_parents = sorted_parents[:, step] # TODO
            step_parent_indicators = batch_indicator(step_parents)

            step_target_iou_xb = tf.gather_nd(params=iou_xb, indices=step_target_indicators)
            step_target_child_sum_hs = tf.gather_nd(params=child_sum_hs, indices=step_target_indicators)
            step_target_gated_child_sum_cs = tf.gather_nd(params=gated_child_sum_cs, indices=step_target_indicators)
            step_target_hs, step_target_cs = aggregate_nodes(step_target_iou_xb, step_target_child_sum_hs, step_target_gated_child_sum_cs)

            step_parent_f_xb = tf.gather_nd(params=f_xb, indices=step_parent_indicators)
            step_gated_memory = gate_memory(step_target_hs, step_target_cs, step_parent_f_xb)

            step_output_mask = sorted_output_mask[:,step] # TODO
            step_propagete_parent_mask = sorted_propagete_parent_mask[:, step] # TODO
            next_hs = hs + tf.scatter_nd(indices=step_target_indicators, updates=step_target_hs*step_output_mask, shape=output_shape) # TODO: tf.where
            next_child_sum_hs = child_sum_hs + tf.scatter_nd(indices=step_parent_indicators, updates=step_target_hs*step_propagete_parent_mask, shape=output_shape)
            next_gated_child_sum_cs = gated_child_sum_cs + tf.scatter_nd(indices=step_parent_indicators, updates=step_gated_memory*step_propagete_parent_mask, shape=output_shape)
            next_step = step + 1
            return next_step, next_hs, next_child_sum_hs, next_gated_child_sum_cs
        def cond(step, hs, child_sum_hs, gated_child_sum_cs):
            return tf.less(step, max_step)
        init_step = tf.constant(0, dtype=max_step.dtype)
        init_hs = tf.zeros(output_shape, dtype=dtype)
        init_child_sum_hs = tf.zeros(output_shape, dtype=dtype)
        init_gated_child_sum_cs = tf.zeros(output_shape, dtype=dtype)
        final_step, hs, child_sum_hs, gated_child_sum_cs = tf.while_loop(cond, loop, [init_step, init_hs, init_child_sum_hs, init_gated_child_sum_cs])

        return hs

    def compute_output_shape(self, input_shape):
        input_shape = tf.TensorShape(input_shape)
        input_shape = input_shape.with_rank(3)
        if input_shape[2].value is None:
            raise ValueError("The last dimension of the inputs must be defined: {}".format(input_shape))
        return input_shape[:2].concatenate(self.units)

    def get_config(self):
        config = {
            "units": self.units,
            }
        base_config = super(BottomUpChildSumTreeLSTM, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


