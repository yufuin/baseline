import contextlib
import tensorflow as tf

class AverageManager:
    def __init__(self, var_list, moving_average=None):
        self.state = "temporal"
        self.dtype = tf.float32
        self.use_moving_average = moving_average is not None
        if self.use_moving_average:
            assert 0.0 < moving_average < 1.0
            self.moving_average = moving_average

        self.var_list = var_list
        self.average_var_list = [tf.Variable(tf.zeros_like(var), name=var.name.split(":")[0]+"/Average", trainable=False) for var in self.var_list]
        self.cache_var_list = [tf.Variable(tf.zeros_like(var), name=var.name.split(":")[0]+"/Average_Cache", trainable=False) for var in self.var_list]
        self.num_updated = tf.Variable(0, name="Average_NumUpdated", trainable=False)

        update_ops = [tf.assign(average_var, new_update) for average_var, new_update in zip(self.average_var_list, self._get_updates())]
        cache_ops = [tf.assign(cache_var, temporal_var) for cache_var, temporal_var in zip(self.cache_var_list, self.var_list)]
        restore_ops = [tf.assign(temporal_var, cache_var) for cache_var, temporal_var in zip(self.cache_var_list, self.var_list)]
        use_average_ops = [tf.assign(temporal_var, average) for temporal_var, average in zip(self.var_list, self._get_averages())]

        # first, update average_var. then update num_updated
        with tf.control_dependencies(update_ops):
            self._update_op = tf.group(tf.assign_add(self.num_updated, 1)) # tf.group to convert into tf.Operation
        self._cache_op = tf.group(cache_ops)
        self._restore_op = tf.group(restore_ops)
        self._use_average_op = tf.group(use_average_ops)


    def _get_updates(self):
        if self.use_moving_average:
            def get_update(temporal_var, average_var):
                return (1.0-self.moving_average) * temporal_var + average_var * self.moving_average
        else:
            float_num_updated = tf.cast(self.num_updated, self.dtype)
            float_next_num_updated = tf.cast(self.num_updated + 1, self.dtype)
            average_fraction = float_num_updated / float_next_num_updated
            temporal_fraction = 1.0 / float_next_num_updated
            def get_update(temporal_var, average_var):
                return temporal_var * temporal_fraction + average_var * average_fraction
        return [get_update(tmp, ave) for tmp, ave in zip(self.var_list, self.average_var_list)]

    def _get_averages(self):
        if self.use_moving_average:
            moving_average_bias = tf.maximum(1.0 - tf.pow(self.moving_average, tf.cast(self.num_updated, self.dtype)), 1e-20)
            return [average_var / moving_average_bias for average_var in self.average_var_list]
        else:
            return self.average_var_list

    @contextlib.contextmanager
    def average_context(self, session=None):
        self.use_average(session=session)
        try:
            yield
        finally:
            self.use_temporal(session=session)

    def update(self, session=None):
        self._update_op.run(session=session)
        return
    def use_average(self, session=None):
        assert self.state == "temporal"
        self.state = "average"
        self._cache_op.run(session=session)
        self._use_average_op.run(session=session)
        return
    def use_temporal(self, session=None):
        assert self.state == "average"
        self.state = "temporal"
        self._restore_op.run(session=session)
        return
