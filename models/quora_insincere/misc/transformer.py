import sys, os
import math
import collections
import re
import multiprocessing
import time
import contextlib
import tqdm
def ctqdm(*args, **kwargs): return contextlib.closing(tqdm.tqdm(*args, **kwargs))

import nltk
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import StratifiedKFold

MODE = 1
INIT_LR = 0.0001
HALF_EPOCH = 300

if MODE == 0:
    BATCH_SIZE = 512
else:
    #BATCH_SIZE = 128
    BATCH_SIZE = 512

MAX_SEQ_LEN = 70

SUBMIT = False
EMBEDDING = "mean"
ENSEMBLE = "pred"

NUM_K_FOLD = 5
EPOCH = 30
AVERAGE = 0.0 # 0.0 => no-average, 1.0 => naive average, 0.0<value<1.0 => moving average

class Parameter:
    use_pos_tag = False
    use_homebrew = False
    finetune = False
    batch_size = BATCH_SIZE
    cell_size1 = 128
    cell_size2 = 128
    key_normalize = False
    num_head = 1
    num_hidden_head = 0
    assert num_hidden_head <= 0 or cell_size2 > 0
    dim_hidden = 32
    use_skip1 = False
    use_skip2 = False
    label_smoothing = 0
    epoch = EPOCH
    lm_epoch = 0
    dim_lm_hidden = 512
    use_max_pool = True
    use_hidden_max_pool = False
    assert not use_hidden_max_pool or cell_size2 > 0
    use_average_pool = False
    use_hidden_average_pool = True
    assert not use_hidden_average_pool or cell_size2 > 0
    parameter_average = AVERAGE # 0.0 => no-average, 1.0 => naive average, 0.0<value<1.0 => moving average

    character_feature = "NONE" # ["NONE", "SumBOC", "SumCharCNN", "ConcatCharCNN"]
    assert character_feature != "SumCharCNN"
    dim_char = 16
    dim_char_out = 32
    char_cnn_window_size = 5


    placeholder_keep_prob_input = tf.placeholder(tf.float32, [])
    placeholder_keep_prob_output = tf.placeholder(tf.float32, [])
    def __init__(self):
        self.keep_prob_input = 0.6
        self.keep_prob_output = 1.0
    def feed(self, wrap=None):
        out = dict() if wrap is None else dict(wrap)
        out[self.placeholder_keep_prob_input] = self.keep_prob_input
        out[self.placeholder_keep_prob_output] = self.keep_prob_output
        return out
    def name(self, suffix=None):
        stack = []
        if self.lm_epoch > 0:
            stack.append("lm{}".format(self.lm_epoch))
        if self.character_feature == "SumBOC":
            stack.append("SumBOC")
        if self.character_feature == "SumCharCNN":
            stack.append("SumCharCNN{}-{}".format(self.dim_char, self.char_cnn_window_size))
        if self.character_feature == "ConcatCharCNN":
            stack.append("ConcatCharCNN{}-{}-{}".format(self.dim_char, self.char_cnn_window_size, self.dim_char_out))
        if self.use_pos_tag:
            stack.append("POS")
        if self.use_homebrew:
            stack.append("homebrew")
        if self.finetune:
            stack.append("finetune")
        stack.append("batch{batch_size}".format(batch_size=self.batch_size))
        if self.cell_size2 > 0:
            stack.append("lstm{cell_size1}-{cell_size2}".format(cell_size1=self.cell_size1, cell_size2=self.cell_size2))
        else:
            stack.append("lstm{cell_size1}".format(cell_size1=self.cell_size1))
        if self.key_normalize:
            stack.append("keynorm")
        stack.append("head{num_head}".format(num_head=self.num_head))
        if self.num_hidden_head > 0:
            stack.append("hhead{num_hidden_head}".format(num_hidden_head=self.num_hidden_head))
        if self.use_max_pool:
            stack.append("maxpool")
        if self.use_hidden_max_pool:
            stack.append("hmaxpool")
        if self.use_average_pool:
            stack.append("avepool")
        if self.use_hidden_average_pool:
            stack.append("havepool")
        stack.append("hidden{dim_hidden}".format(dim_hidden=self.dim_hidden))
        stack.append("ki{ki}-ko{ko}".format(ki=self.keep_prob_input, ko=self.keep_prob_output))
        if any([self.use_skip1, self.use_skip2]):
            skips = [n for n,f in (("1", self.use_skip1), ("2", self.use_skip2)) if f]
            stack.append("skip" + "-".join(skips))
        if self.label_smoothing > 0.0:
            stack.append("smooth{}".format(self.label_smoothing))
        if self.parameter_average > 0.0:
            if self.parameter_average == 1.0:
                stack.append("pnave")
            else:
                stack.append("pmave{}".format(self.parameter_average))

        if suffix is not None and len(suffix) > 0:
            stack.append(suffix)
        return "_".join(stack)

params = Parameter()


train_df = pd.read_csv("../input/train.csv")

preprocessed = np.load("preprocessed.npy").tolist()

id_to_word = preprocessed["id_to_word"]

if EMBEDDING == "glove":
    init_emb = preprocessed["glove_emb"]
elif EMBEDDING == "paragram":
    init_emb = preprocessed["paragram_emb"]
elif EMBEDDING == "mean":
    init_emb = preprocessed["mean_emb"]
else:
    raise ValueError()

if params.use_homebrew:
    homebrew_init_emb = preprocessed["homebrew_init_emb"]

if params.use_pos_tag:
    id_to_pos_tag = preprocessed["id_to_pos_tag"]

if params.character_feature != "NONE":
    id_to_char = preprocessed["id_to_char"]

if params.lm_epoch > 0:
    initial_char_set = {word[0] for word in id_to_word}
    initial_char_to_id = {char:(2+id_) for id_, char in enumerate(sorted(initial_char_set))}
    initial_char_set.add("$$UNK$$")
    initial_char_to_id["$$UNK$$"] = 0
    initial_char_set.add("$$BOS/EOS$$")
    initial_char_to_id["$$BOS/EOS$$"] = 1
    word_id_to_initial_char_id = {word_id:initial_char_to_id[word[0]] for word_id, word in enumerate(id_to_word)}


all_train_instances = preprocessed["all_train_instances"]
test_instances = preprocessed["test_instances"]
def pick(instance):
    new = dict()
    new["word"] = instance["word"]
    new["sequence_length"] = instance["sequence_length"]
    new["label"] = instance["label"]
    new["index"] = instance["index"]
    return new
all_train_instances = np.array([pick(instance) for instance in all_train_instances])
test_instances = np.array([pick(instance) for instance in test_instances])

all_train_labels = [instance["label"] for instance in all_train_instances]
skfold = StratifiedKFold(NUM_K_FOLD, shuffle=True, random_state=777)
cv_splits = list(skfold.split(all_train_labels, all_train_labels))

def pad(sequences, atom):
    assert type(sequences[0]) in [list, tuple]
    max_len = max(len(sequence) for sequence in sequences)
    return np.stack([list(sequence) + [atom]*(max_len-len(sequence)) for sequence in sequences], axis=0)
class Gen:
    def __init__(self, all_instances, batch_size, shuffle):
        self.all_instances = all_instances
        self.batch_size = batch_size
        self.shuffle = shuffle
    def construct(self, indices=None):
        if indices is None:
            target = self.all_instances
        else:
            target = self.all_instances[indices]

        sorter = sorted(np.arange(len(target)), key=lambda x:target[x]["sequence_length"])
        self.target = target[sorter]

        keys = list(self.target[0].keys())
        chars_atom = None
        self.backets = []
        for b in range(0, len(self.target), self.batch_size):
            backet = {}
            for key in keys:
                value = [instance[key] for instance in self.target[b:b+self.batch_size]]
                if type(value[0]) in [list, tuple]:
                    if key == "chars":
                        if params.character_feature == "NONE": continue
                        if chars_atom is None:
                            chars_atom = [0] * len(value[0][0])
                        value = pad(value, atom=chars_atom)
                    else:
                        value = pad(value, atom=0)
                backet[key] = value

                if params.lm_epoch > 0:
                    key = "lm_gold"
                    value = np.vectorize(lambda x: word_id_to_initial_char_id[x])(backet["word"])
                    zero_line = np.ones_like(value[:,0:1])
                    value = np.concatenate([zero_line, value, zero_line], axis=1)
                    backet[key] = value

            self.backets.append(backet)
        return self
    @property
    def output_shapes(self):
        sample = self.all_instances[0]
        outs = dict()
        for key, value in sample.items():
            if type(value) in [list, tuple]:
                if key == "chars":
                    if params.character_feature == "NONE": continue
                    outs[key] = tf.TensorShape([None, None, len(value[0])])
                else:
                    outs[key] = tf.TensorShape([None, None])
            else:
                outs[key] = tf.TensorShape([None])

        if params.lm_epoch > 0:
            outs["lm_gold"] = tf.TensorShape([None, None])

        return outs
    @property
    def output_types(self):
        sample = self.all_instances[0]
        outs = dict()
        for key, value in sample.items():
            if key == "chars":
                if params.character_feature == "NONE":
                    continue

            while type(value) in [list, tuple]:
                value = value[0]

            if type(value) is float:
                outs[key] = tf.float32
            elif type(value) is int:
                outs[key] = tf.int32
            elif type(value) is str:
                outs[key] = tf.string
            elif value.dtype in [np.float32, np.float64]:
                outs[key] = tf.float32
            elif value.dtype in [np.int32, np.int64]:
                outs[key] = tf.int32

        if params.lm_epoch > 0:
            outs["lm_gold"] = tf.int32

        return outs
    def __call__(self):
        order = np.arange(len(self.backets))
        if self.shuffle:np.random.shuffle(order)
        for o in order:
            yield self.backets[o]
    def get_target_len(self):
        return len(self.target)
train_gen = Gen(all_train_instances, batch_size=params.batch_size, shuffle=True)
val_gen = Gen(all_train_instances, batch_size=params.batch_size, shuffle=False)
test_gen = Gen(test_instances, batch_size=params.batch_size, shuffle=False).construct()

train_dataset = tf.data.Dataset.from_generator(train_gen, output_shapes=train_gen.output_shapes, output_types=train_gen.output_types)
val_dataset = tf.data.Dataset.from_generator(val_gen, output_shapes=val_gen.output_shapes, output_types=val_gen.output_types)
test_dataset = tf.data.Dataset.from_generator(test_gen, output_shapes=test_gen.output_shapes, output_types=test_gen.output_types)
train_dataset = train_dataset.prefetch(8)
val_dataset = val_dataset.prefetch(8)
test_dataset = test_dataset.prefetch(8)
dataset_iterator = tf.data.Iterator.from_structure(output_shapes=train_dataset.output_shapes, output_types=train_dataset.output_types)
init_train = dataset_iterator.make_initializer(train_dataset)
init_val = dataset_iterator.make_initializer(val_dataset)
init_test = dataset_iterator.make_initializer(test_dataset)
minibatch = dataset_iterator.get_next()

word_ids = minibatch.get("word")[:,:MAX_SEQ_LEN]
seq_lens = minibatch.get("sequence_length")
labels = minibatch.get("label")
instance_indices = minibatch.get("index")


is_training = tf.placeholder(tf.bool, [])
def dropout(target, keep_prob):
    return tf.cond(is_training, lambda: tf.nn.dropout(target, keep_prob=keep_prob), lambda: target)
def add_noise(target, stddev=0.1):
    return tf.cond(is_training, lambda: target + tf.random_normal(tf.shape(target), stddev=stddev), lambda: target)

class AverageManager:
    def __init__(self, var_list, moving_average=None):
        self.state = "temporal"

        self.use_moving_average = moving_average is not None
        if self.use_moving_average:
            assert 0.9 <= moving_average < 1.0
            self.moving_average = moving_average

        self.num_updated = tf.Variable(0, name="Average_NumUpdated", trainable=False)
        self.increase_num_updated = tf.assign_add(self.num_updated, 1)

        float_num_updated = tf.to_float(self.num_updated)
        float_next_num_updated = tf.to_float(self.num_updated + 1)
        average_fraction = float_num_updated / float_next_num_updated
        temporal_fraction = 1.0 / float_next_num_updated
        if self.use_moving_average:
            moving_average_bias = tf.maximum(1.0 - tf.pow(self.moving_average, float_num_updated), 1e-20)

        self.update_ops = []
        self.store_ops = []
        self.restore_ops = []
        self.switch_ops = []

        self.var_list = var_list
        for var in self.var_list:
            averaged = tf.Variable(tf.zeros_like(var), name=var.name.split(":")[0]+"/Average", trainable=False)
            cache = tf.Variable(tf.zeros_like(var), name=var.name.split(":")[0]+"/Average_Cache", trainable=False)

            if self.use_moving_average:
                new_average = averaged * self.moving_average + var * (1.0-self.moving_average)
            else:
                new_average = averaged * average_fraction + var * temporal_fraction
            update_op = tf.assign(averaged, new_average)
            self.update_ops.append(update_op)

            store_tmp_to_cache = tf.assign(cache, var)
            restore_tmp = tf.assign(var, cache)
            self.store_ops.append(store_tmp_to_cache)
            self.restore_ops.append(restore_tmp)

            if self.use_moving_average:
                switch_to_average = tf.assign(var, averaged / moving_average_bias)
            else:
                switch_to_average = tf.assign(var, averaged)
            self.switch_ops.append(switch_to_average)

    def update_averages(self, sess):
        sess.run(self.update_ops)
        sess.run(self.increase_num_updated)
        return
    def switch_to_average(self, sess):
        assert self.state == "temporal"
        self.state = "average"
        sess.run(self.store_ops)
        sess.run(self.switch_ops)
        return
    def switch_to_temporal(self, sess):
        assert self.state == "average"
        self.state = "temporal"
        sess.run(self.restore_ops)
        return

def layer_normalize(vec, axis):
    mean, var = tf.nn.moments(vec, axis)
    return (vec - tf.expand_dims(mean, axis)) / tf.expand_dims(tf.maximum(tf.sqrt(var), 1e-10), axis)
class MultiHeadReduction:
    def __init__(self, num_head, dim_input, key_normalize, position_type="none", dim_position=15):
        assert dim_input % num_head == 0
        assert position_type in ["none", "concat", "add"]

        self.num_head = num_head
        self.dim_input = dim_input

        self.key_normalize = key_normalize

        self.position_type = str(position_type)
        if self.position_type == "concat":
            raise NotImplementedError("MultiHeadReduction(position_type=concat)")
            self.dim_position = dim_position
        elif self.position_type == "add":
            raise NotImplementedError("MultiHeadReduction(position_type=add)")
            self.dim_position = 0
        else:
            assert self.position_type == "none"
            self.dim_position = 0

        with tf.variable_scope("MultiHead"):
            self.w_kvs = tf.get_variable("w_kvs", [self.dim_input+self.dim_position, 2*self.dim_input])
            self.w_queries = tf.get_variable("w_queries", [self.num_head, self.dim_input//self.num_head])
            self.root_d_k = tf.constant(np.sqrt(self.dim_input//self.num_head), dtype=tf.float32)
    def __call__(self, inputs, seq_lens, batch_size=None, max_seq_len=None):
        if batch_size is None: batch_size = tf.shape(inputs)[0]
        if max_seq_len is None: max_seq_len = tf.shape(inputs)[1]
        seq_mask = tf.sequence_mask(seq_lens, max_seq_len, dtype=tf.float32)[:,:,tf.newaxis,tf.newaxis]
        flat_inputs = tf.reshape(inputs, [batch_size*max_seq_len, self.dim_input])
        flat_concat_kvs = tf.matmul(flat_inputs, self.w_kvs)
        kvs = tf.reshape(flat_concat_kvs, [batch_size, max_seq_len, 2*self.num_head, self.dim_input//self.num_head])
        keys, values = tf.split(kvs, 2, axis=2) # 2x[batch_size, seq_len, num_head, dim]
        queries = self.w_queries
        if self.key_normalize:
            keys = layer_normalize(keys, -1)
        us = tf.clip_by_value(tf.reduce_sum(keys*queries, axis=3, keepdims=True) / self.root_d_k, -20.0, 20.0)
        exp_us = tf.exp(us) * seq_mask # [batch_size, seq_len, num_head, 1]
        attentions = exp_us / (tf.reduce_sum(exp_us, axis=1, keepdims=True) + 1e-10) # [batch_size, seq_len, num_head, 1]
        reduction = tf.reduce_sum(tf.reshape(values * attentions, [batch_size, max_seq_len, self.dim_input]), axis=1) # [batch_size, num_head*dim]
        self.attentions = tf.squeeze(attentions, axis=3) # [batch_size,seq_len,num_head]
        return reduction

def position_emb(dim):
    assert dim % 2 == 0
    pos = np.arange(MAX_SEQ_LEN, dtype=np.float32) # [seq_len]
    d = np.arange(dim//2, dtype=np.float32)
    div = np.power(10000.0, d/(dim//2)) # [dim//2]
    theta = pos[:,np.newaxis] / div[np.newaxis,:] # [seq_len, dim//2]
    emb_cos = np.cos(theta)
    emb_sin = np.sin(theta)
    emb = np.concatenate([emb_cos, emb_sin], axis=1) # [seq_len, dim]
    return emb

class MultiHeadTransformer:
    def __init__(self, num_head, dim_input, dim_output, key_normalize, position_type="add"):
        assert dim_output % num_head == 0
        assert dim_input % 2 == 0
        assert position_type in ["none", "add"]

        self.num_head = num_head
        self.dim_output = dim_output
        self.dim_input = dim_input

        self.key_normalize = key_normalize

        self.position_type = str(position_type)
        if self.position_type == "add":
            self.position_emb = tf.constant(position_emb(self.dim_input), dtype=tf.float32)
            self.w_position = tf.Variable(1.0, name="w_position")

        initializer = tf.glorot_uniform_initializer()
        self.w_kvqs = tf.Variable(initializer([self.dim_input, 3*self.dim_output]), name="w_kvqs")
        self.root_d_k = tf.constant(np.sqrt(self.dim_output//self.num_head), dtype=tf.float32)
    def __call__(self, inputs, seq_lens, batch_size=None, max_seq_len=None):
        if batch_size is None: batch_size = tf.shape(inputs)[0]
        if max_seq_len is None: max_seq_len = tf.shape(inputs)[1]
        seq_mask = tf.sequence_mask(seq_lens, max_seq_len, dtype=tf.float32)
        if self.position_type == "add":
            inputs = inputs + self.position_emb[tf.newaxis, :max_seq_len]
            #inputs = inputs + self.w_position * self.position_emb[tf.newaxis, :max_seq_len]
        flat_inputs = tf.reshape(inputs, [batch_size*max_seq_len, self.dim_input])
        flat_concat_kvqs = tf.matmul(flat_inputs, self.w_kvqs)
        kvqs = tf.reshape(flat_concat_kvqs, [batch_size, max_seq_len, 3*self.num_head, self.dim_output//self.num_head])
        keys, values, queries = tf.split(kvqs, 3, axis=2) # 3x[batch_size, seq_len, num_head, dim]
        if self.key_normalize:
            keys = layer_normalize(keys, -1)
        us = tf.clip_by_value(tf.reduce_sum(keys[:,tf.newaxis]*queries[:,:,tf.newaxis], axis=-1, keepdims=True) / self.root_d_k, -20.0, 20.0)
        exp_us = tf.exp(us) * seq_mask[:,tf.newaxis,:,tf.newaxis,tf.newaxis] # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        attentions = exp_us / (tf.reduce_sum(exp_us, axis=2, keepdims=True) + 1e-10) # [batch_size, query_seq_len, key_seq_len, num_head, 1]
        reduction = tf.reshape(tf.reduce_sum(values[:,tf.newaxis] * attentions, axis=2), [batch_size, max_seq_len, self.dim_output])
        self.attentions = tf.squeeze(attentions, axis=-1) # [batch_size,query_seq_len,key_seq_len,num_head]
        return reduction # [batch_size, max_seq_len, dim_output]

class Model:
    dim_word_emb = 300
    dim_pos_tag = 30
    num_unit1 = params.cell_size1
    num_unit2 = params.cell_size2
    num_head = params.num_head
    num_hidden_head = params.num_hidden_head
    dim_fc1 = params.dim_hidden
    use_skip1 = params.use_skip1
    use_skip2 = params.use_skip2
    def __init__(self):
        self.w_word_emb = tf.get_variable("w_word_emb", init_emb.shape, initializer=tf.constant_initializer(init_emb), trainable=params.finetune)
        if params.use_pos_tag:
            self.w_pos_tag_emb = tf.get_variable("w_pos_tag_emb", [len(id_to_pos_tag), self.dim_pos_tag], initializer=tf.random_uniform_initializer())
        if params.use_homebrew:
            self.w_homebrew_word_emb = tf.get_variable("w_homebrew_word_emb", homebrew_init_emb.shape, initializer=tf.constant_initializer(homebrew_init_emb), trainable=params.finetune)
        #self.flstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit1)
        #self.blstm1 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit1)
        self.flstm1 = tf.keras.layers.CuDNNLSTM(self.num_unit1, return_sequences=True)
        self.blstm1 = tf.keras.layers.CuDNNLSTM(self.num_unit1, return_sequences=True)

        if self.use_skip1:
            self.w_skip1 = tf.get_variable("w_skip1", [int(init_emb.shape[-1]), 2*self.num_unit1])
        if self.num_unit2 > 0:
            #self.flstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit2)
            #self.blstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit2)
            self.flstm2 = tf.keras.layers.CuDNNLSTM(self.num_unit2, return_sequences=True)
            self.blstm2 = tf.keras.layers.CuDNNLSTM(self.num_unit2, return_sequences=True)
            if self.use_skip2:
                self.w_skip2 = tf.get_variable("w_skip2", [2*self.num_unit1, 2*self.num_unit2])
            self.multi1 = MultiHeadReduction(num_head=self.num_head, dim_input=2*self.num_unit2, key_normalize=params.key_normalize)
        else:
            self.multi1 = MultiHeadReduction(num_head=self.num_head, dim_input=2*self.num_unit1, key_normalize=params.key_normalize)
        self.fc_output = tf.keras.layers.Dense(1)

        word_emb = tf.nn.embedding_lookup(params=self.w_word_emb, ids=word_ids) # [batch_size, seq_len, dim_word_emb]

        word_vec = word_emb

        #"SumCharCNN"
        if params.character_feature == "SumBOC":
            self.w_char_emb = tf.get_variable("w_char_emb", [len(id_to_char), init_emb.shape[-1]], initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=True)
            char_ids = minibatch.get("chars") # [batch_size, seq_len, word_len]
            char_emb = tf.nn.embedding_lookup(params=self.w_char_emb, ids=char_ids) # [batch_size, seq_len, word_len, dim_word_emb]
            char_mask = tf.cast(tf.not_equal(char_ids, 0), tf.float32) # [batch_size, seq_len, word_len]
            char_vec = tf.reduce_sum(char_emb * char_mask[:,:,:,tf.newaxis], axis=2) # [batch_size, seq_len, dim_word_emb]
            word_vec = word_vec + char_vec
        elif params.character_feature == "ConcatCharCNN":
            self.w_char_emb = tf.get_variable("w_char_emb", [len(id_to_char), params.dim_char], initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=True)
            self.char_cnn = tf.keras.layers.Conv1D(params.dim_char_out, params.char_cnn_window_size, padding="same")
            char_ids = minibatch.get("chars") # [batch_size, seq_len, word_len]
            char_emb = tf.nn.embedding_lookup(params=self.w_char_emb, ids=char_ids) # [batch_size, seq_len, word_len, dim_char_emb]
            char_shapes = tf.shape(char_emb)
            char_shapes = [char_shapes[i] for i in range(4)]
            flat_char_emb = tf.reshape(char_emb, [char_shapes[0]*char_shapes[1], char_shapes[2], params.dim_char])
            flat_h_char_cnn = tf.nn.sigmoid(self.char_cnn(flat_char_emb))
            char_mask = tf.cast(tf.not_equal(char_ids, 0), tf.float32) # [batch_size, seq_len, word_len]
            h_char_cnn = tf.reshape(flat_h_char_cnn, char_shapes[:3]+[params.dim_char_out]) * char_mask[:,:,:,tf.newaxis]
            max_h_char_cnn = tf.reduce_max(h_char_cnn, axis=2)
            mean_h_char_cnn = tf.reduce_sum(h_char_cnn, axis=2) / (tf.reduce_sum(char_mask, axis=2)[:,:,tf.newaxis] + 1e-10)
            word_vec = tf.concat([word_vec, max_h_char_cnn, mean_h_char_cnn], axis=2)

        if params.use_homebrew:
            homebrew_word_emb = tf.nn.embedding_lookup(params=self.w_homebrew_word_emb, ids=minibatch["homebrew_word"]) # [batch_size, seq_len, dim_word_emb]
            word_vec = tf.concat([word_emb, homebrew_word_emb], axis=2)

        if params.use_pos_tag:
            pos_tag_emb = tf.nn.embedding_lookup(params=self.w_pos_tag_emb, ids=minibatch["pos"]) # [batch_size, seq_len, dim_word_emb]
            word_vec = tf.concat([word_vec, pos_tag_emb], axis=2)

        def reverse_sequence(inputs):
            #return tf.reverse_sequence(inputs, seq_lens, seq_axis=0, batch_axis=1)
            return tf.reverse_sequence(inputs, seq_lens, batch_axis=0, seq_axis=1)
        def bilstm(inputs, flstm, blstm, return_all=False):
            #h_flstm, _ = flstm(inputs)
            #h_blstm, _ = blstm(reverse_sequence(inputs))
            h_flstm = flstm(inputs)
            h_blstm = blstm(reverse_sequence(inputs))
            h_blstm = reverse_sequence(h_blstm)
            if return_all:
                return tf.concat([h_flstm, h_blstm], axis=2), h_flstm, h_blstm
            else:
                return tf.concat([h_flstm, h_blstm], axis=2)

        """
        #word_vec = tf.transpose(word_vec, [1,0,2])
        h_bilstm1 = bilstm(word_vec, self.flstm1, self.blstm1)
        if self.use_skip1:
            prev = word_vec
            weight = self.w_skip1

            shape = tf.shape(prev)
            ndim = len(prev.shape)
            flat_prev = tf.reshape(prev, [-1, shape[-1]])
            skip = tf.reshape(tf.matmul(flat_prev, weight), [shape[i] for i in range(ndim-1)] + [weight.shape[-1]])

            h_bilstm1 = h_bilstm1 + skip

        if self.num_unit2 > 0:
            h_bilstm2 = bilstm(h_bilstm1, self.flstm2, self.blstm2)
            if self.use_skip2:
                prev = h_bilstm1
                weight = self.w_skip2

                shape = tf.shape(prev)
                ndim = len(prev.shape)
                flat_prev = tf.reshape(prev, [-1, shape[-1]])
                skip = tf.reshape(tf.matmul(flat_prev, weight), [shape[i] for i in range(ndim-1)] + [weight.shape[-1]])

                h_bilstm2 = h_bilstm2 + skip
            #h_bilstm = tf.transpose(h_bilstm2, [1,0,2])
            h_bilstm = h_bilstm2
        else:
            #h_bilstm = tf.transpose(h_bilstm1, [1,0,2])
            h_bilstm = h_bilstm1

        input_fc1 = []

        input_fc1.append(self.multi1(h_bilstm, seq_lens=seq_lens))

        if self.num_hidden_head > 0:
            with tf.variable_scope("hidden_multi"):
                self.hidden_multi = MultiHeadReduction(num_head=self.num_hidden_head, dim_input=2*self.num_unit1, key_normalize=params.key_normalize)
            input_fc1.append(self.hidden_multi(h_bilstm1, seq_lens=seq_lens))

        h_bilstm_mask = tf.cast(tf.sequence_mask(seq_lens, tf.shape(word_ids)[1]), tf.float32)[:,:,tf.newaxis]
        masked_h_bilstm = h_bilstm * h_bilstm_mask
        masked_h_bilstm1 = h_bilstm1 * h_bilstm_mask
        def max_pool(masked_h_sequence, mask):
            min_value = tf.reduce_min(masked_h_sequence, axis=1) # [batch_size, dim]
            biased = masked_h_sequence - min_value[:,tf.newaxis] # all non-masked values are at least 0, but masked values could be over 0
            masked_biased = biased * mask # all non-masked values are at least 0, while masked values must be 0.
            max_pooled_biased = tf.reduce_max(masked_biased, axis=1)
            max_pooled = max_pooled_biased + min_value
            return max_pooled
        if params.use_max_pool:
            input_fc1.append(max_pool(masked_h_bilstm, h_bilstm_mask))
        if params.use_hidden_max_pool:
            input_fc1.append(max_pool(masked_h_bilstm1, h_bilstm_mask))
        if params.use_average_pool:
            average_pooled = tf.reduce_sum(masked_h_bilstm, axis=1) / (tf.cast(seq_lens, tf.float32)+1e-10)[:,tf.newaxis]
            input_fc1.append(average_pooled)
        if params.use_hidden_average_pool:
            average_pooled = tf.reduce_sum(masked_h_bilstm1, axis=1) / (tf.cast(seq_lens, tf.float32)+1e-10)[:,tf.newaxis]
            input_fc1.append(average_pooled)

        if len(input_fc1) == 1:
            input_fc1 = input_fc1[0]
        else:
            input_fc1 = tf.concat(input_fc1, axis=-1)
        """

        if MODE == 0:
            word_vec = dropout(word_vec, params.keep_prob_input)
            h = word_vec

            h = bilstm(h, self.flstm1, self.blstm1)
            h = MultiHeadTransformer(num_head=2, dim_input=256, dim_output=256, key_normalize=False, position_type="add")(h, seq_lens)
            input_fc1 = self.multi1(h, seq_lens=seq_lens)
            self.fc1 = tf.keras.layers.Dense(self.dim_fc1)
            h_fc1 = tf.nn.tanh(self.fc1(input_fc1))

        else:
            res_dim = 256
            wide_dim = 1024

            word_vec = dropout(word_vec, params.keep_prob_input)
            h = word_vec
            l = tf.keras.layers.Dense(res_dim, use_bias=False)
            h = l(h)

            def residual(func, x):
                return layer_normalize(func(x) + x, -1)
                #return func(x) + x
                #return (func(x) + x) / tf.sqrt(2.0)


            self.attentions = []
            for _ in range(3):
                l = MultiHeadTransformer(num_head=8, dim_input=res_dim, dim_output=res_dim, key_normalize=False, position_type="add")
                h = residual(lambda x: l(x, seq_lens=seq_lens), h)
                self.attentions.append(l.attentions)

                l1 = tf.keras.layers.Dense(wide_dim)
                l2 = tf.keras.layers.Dense(res_dim)
                h = residual(lambda x: l2(tf.nn.leaky_relu(l1(x))), h)

                l = MultiHeadTransformer(num_head=8, dim_input=res_dim, dim_output=res_dim, key_normalize=False, position_type="add")
                h = residual(lambda x: l(x, seq_lens=seq_lens), h)
                self.attentions.append(l.attentions)

                l1 = tf.keras.layers.Dense(wide_dim)
                l2 = tf.keras.layers.Dense(res_dim)
                h = residual(lambda x: l2(tf.nn.leaky_relu(l1(x))), h)

            with tf.variable_scope("out"):
                l = MultiHeadReduction(num_head=8, dim_input=res_dim, key_normalize=False)
                input_fc1 = l(h, seq_lens=seq_lens)
                self.attentions.append(l.attentions)
            self.fc1 = tf.keras.layers.Dense(128)
            h_fc1 = tf.nn.tanh(self.fc1(input_fc1))

        #h_fc1 = add_noise(h_fc1, 0.1)
        h_fc1 = dropout(h_fc1, params.keep_prob_output)
        self.logits = self.fc_output(h_fc1)
print("build model")
model = Model()
logits = model.logits
loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels[:,tf.newaxis], logits=logits, label_smoothing=params.label_smoothing)
mean_loss, update_mean_loss = tf.metrics.mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels[:,tf.newaxis], logits=logits, reduction=tf.losses.Reduction.NONE))
adam_lr = tf.Variable(INIT_LR, trainable=False)
half_adam_lr = adam_lr.assign(adam_lr/2.0)
opt = tf.train.AdamOptimizer(learning_rate=adam_lr)
train_op = opt.minimize(loss)

if params.lm_epoch > 0:
    lm_f_logits, lm_b_logits = model.lm_logits
    lm_gold = minibatch["lm_gold"]
    lm_f_gold = lm_gold[:, 2:] # [batch_size, seq_len]
    lm_b_gold = lm_gold[:, :-2]
    lm_loss_mask = tf.cast(tf.sequence_mask(seq_lens, tf.shape(word_ids)[1]), tf.float32) # [batch_size, seq_len]
    lm_f_loss = tf.losses.sparse_softmax_cross_entropy(labels=lm_f_gold, logits=lm_f_logits, weights=lm_loss_mask)
    lm_b_loss = tf.losses.sparse_softmax_cross_entropy(labels=lm_b_gold, logits=lm_b_logits, weights=lm_loss_mask)
    lm_loss = lm_f_loss + lm_b_loss
    lm_train_op = opt.minimize(lm_loss)

if params.parameter_average > 0.0:
    if params.parameter_average == 1.0:
        average_manager = AverageManager(tf.trainable_variables())
    else:
        average_manager = AverageManager(tf.trainable_variables(), moving_average=params.parameter_average)

probs = tf.nn.sigmoid(logits)[:,0]
preds = tf.to_int32(tf.greater(logits, 0.0))[:,0]
accuracy, update_accuracy = tf.metrics.accuracy(predictions=preds, labels=labels)
recall, update_recall = tf.metrics.accuracy(predictions=preds, labels=labels, weights=tf.to_float(tf.equal(labels, 1)), name="recall")
precision, update_precision = tf.metrics.accuracy(predictions=preds, labels=labels, weights=tf.to_float(tf.equal(preds, 1)), name="precision")
f_score = recall * precision * 2.0 / (1e-10+recall+precision)
upds = [update_recall, update_precision, update_accuracy]
print("built tensors")

print("summary")
epoch_summaries = []
epoch_summaries.append(tf.summary.scalar("loss", mean_loss))
epoch_summaries.append(tf.summary.scalar("accuracy", accuracy))
epoch_summaries.append(tf.summary.scalar("precision", precision))
epoch_summaries.append(tf.summary.scalar("recall", recall))
epoch_summaries.append(tf.summary.scalar("f-score", f_score))
val_epoch_summaries = []
placeholder_threshold = tf.placeholder(tf.float32, [])
val_epoch_summaries.append(tf.summary.scalar("threshold", placeholder_threshold))
placeholder_balanced_precision = tf.placeholder(tf.float32, [])
val_epoch_summaries.append(tf.summary.scalar("balanced_precision", placeholder_balanced_precision))
placeholder_balanced_recall = tf.placeholder(tf.float32, [])
val_epoch_summaries.append(tf.summary.scalar("balanced_recall", placeholder_balanced_recall))
placeholder_balanced_f_score = tf.placeholder(tf.float32, [])
val_epoch_summaries.append(tf.summary.scalar("balanced_f-score", placeholder_balanced_f_score))
val_end_summaries = []
placeholder_best_threshold = tf.placeholder(tf.float32, [])
val_end_summaries.append(tf.summary.scalar("best_threshold", placeholder_best_threshold))
placeholder_best_balanced_precision = tf.placeholder(tf.float32, [])
val_end_summaries.append(tf.summary.scalar("best_balanced_precision", placeholder_best_balanced_precision))
placeholder_best_balanced_recall = tf.placeholder(tf.float32, [])
val_end_summaries.append(tf.summary.scalar("best_balanced_recall", placeholder_best_balanced_recall))
placeholder_best_balanced_f_score = tf.placeholder(tf.float32, [])
val_end_summaries.append(tf.summary.scalar("best_balanced_f-score", placeholder_best_balanced_f_score))

merged_epoch = tf.summary.merge(epoch_summaries)
merged_val_epoch = tf.summary.merge(val_epoch_summaries)
merged_val_end = tf.summary.merge(val_end_summaries)
def train_epoch_summarize(writer, epoch):
    summary = sess.run(merged_epoch)
    writer.add_summary(summary, epoch)
    writer.flush()
def val_epoch_summarize(writer, epoch, threshold, precision, recall, f_score):
    summaries = sess.run([merged_epoch, merged_val_epoch], feed_dict={placeholder_threshold:threshold,
                                                                      placeholder_balanced_precision:precision,
                                                                      placeholder_balanced_recall:recall,
                                                                      placeholder_balanced_f_score:f_score})
    for summary in summaries:
        writer.add_summary(summary, epoch)
    writer.flush()
def val_end_summarize(writer, threshold, precision, recall, f_score, loss=None):
    summary = sess.run(merged_val_end, feed_dict={placeholder_best_threshold:threshold,
                                                  placeholder_best_balanced_precision:precision,
                                                  placeholder_best_balanced_recall:recall,
                                                  placeholder_best_balanced_f_score:f_score})
    writer.add_summary(summary, 0)
    writer.flush()


def do_val(epoch, writer, num_print_failure=0):
    feed = params.feed({is_training:False})
    sess.run(tf.local_variables_initializer())
    if params.parameter_average > 0.0:
        average_manager.switch_to_average(sess)
    sess.run(init_val)
    val_all_probs = []
    val_all_labels = []
    val_all_indices = []

    with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=val_gen.get_target_len())) as pbar:
        while True:
            p, g, l, acc, rec, prec, f, idx = sess.run([probs, labels, update_mean_loss, update_accuracy, update_recall, update_precision, f_score, instance_indices], feed_dict=feed)
            val_all_probs.append(p)
            val_all_labels.append(g)
            val_all_indices.append(idx)
            if pbar.n % 1000 == 0: pbar.set_description("loss:{loss:.4e} acc:{accuracy:.4f} prec:{precision:.4f} rec:{recall:.4f} f:{f_score:.4f}".format(loss=l, accuracy=acc, precision=prec, recall=rec, f_score=f))
            pbar.update(len(p))
    val_all_probs = np.concatenate(val_all_probs, axis=0)
    val_all_labels = np.concatenate(val_all_labels, axis=0)
    val_all_indices = np.concatenate(val_all_indices, axis=0)
    acc, prec, rec, f = sess.run([accuracy, precision, recall, f_score], feed_dict=feed)
    print("val  ", acc, prec, rec, f)

    positive = val_all_labels.sum()
    best_f_score = -1.0
    results = []
    for i in range(0, 100):
        threshold = i / 100.0
        preds_on_threshold = (val_all_probs >= threshold).astype(np.int32)
        positive_correct = (np.equal(preds_on_threshold, val_all_labels).astype(np.int32) * val_all_labels).sum()
        suspect = preds_on_threshold.sum()
        prec = positive_correct / max(suspect, 1)
        rec = positive_correct / max(positive, 1)
        f = prec * rec * 2.0 / (1e-10+prec+rec)
        if f > best_f_score:
            best_f_score = f
            best_threshold = threshold
            best_precision = prec
            best_recall = rec
            best_i = i
        results.append("@{}  f={}  prec={}  rec={}".format(threshold, f, prec, rec))
    for line in results[best_i-5:best_i+5+1]:
        print(line)
    outs = {"f_score":best_f_score, "threshold":best_threshold, "precision":best_precision, "recall":best_recall}
    if writer is not None:
        val_epoch_summarize(writer=writer, epoch=epoch, **outs)
    outs["loss"] = sess.run(mean_loss)

    if num_print_failure > 0:
        false_negative = [(p,g,idx) for p,g,idx in zip(val_all_probs, val_all_labels, val_all_indices) if g==1 and g != int(p>=best_threshold)]
        false_positive = [(p,g,idx) for p,g,idx in zip(val_all_probs, val_all_labels, val_all_indices) if g==0 and g != int(p>=best_threshold)]
        np.random.shuffle(false_negative)
        np.random.shuffle(false_positive)
        for target in [false_negative, false_positive]:
            for p,g,idx in target[:num_print_failure]:
                print(train_df.target[idx], g, p, train_df.question_text[idx])

    if params.parameter_average > 0.0:
        average_manager.switch_to_temporal(sess)
    return outs
def do_train(epoch, writer):
    feed = params.feed({is_training:True})
    sess.run(tf.local_variables_initializer())
    sess.run(init_train)

    with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=train_gen.get_target_len())) as pbar:
        while True:
            p, _, l, acc, rec, prec, f = sess.run([probs, train_op, update_mean_loss, update_accuracy, update_recall, update_precision, f_score], feed_dict=feed)
            if params.parameter_average > 0.0:
                average_manager.update_averages(sess)
            if pbar.n % 1000 == 0: pbar.set_description("loss:{loss:.4e} acc:{accuracy:.4f} prec:{precision:.4f} rec:{recall:.4f} f:{f_score:.4f}".format(loss=l, accuracy=acc, precision=prec, recall=rec, f_score=f))
            pbar.update(len(p))
    if writer is not None:
        train_epoch_summarize(writer, epoch)

def do_attention(epoch):
    feed = params.feed({is_training:False})
    sess.run(tf.local_variables_initializer())
    sess.run(init_train)

    outs = []
    with contextlib.suppress(tf.errors.OutOfRangeError):
        for _ in range(1):
            ws, l, p, g, attentions = sess.run([word_ids, seq_lens, probs, labels, model.attentions], feed_dict=feed)
            attentions = zip(*attentions) # [[batch_size, seq_len, seq_len], ...] => [[seq_len,seq_len],...][batch_size]
            for sp, sg, sattentions, sl, sws in zip(p,g,attentions, l, ws):
                out = {}
                sws = sws[:sl]
                out["word"] = [id_to_word[id_] for id_ in sws]
                out["label"] = int(sg)
                out["prob"] = float(sp)
                for i, att in enumerate(sattentions):
                    out["attention_{}".format(i)] = att.tolist()
                outs.append(out)
    import json
    with open(f"attentions/attention_epoch{epoch:03d}.json", "w") as f:
        json.dump(outs, f)

def do_lm(max_epoch):
    for epoch in range(max_epoch):
        feed = params.feed({is_training:True})
        sess.run(tf.local_variables_initializer())
        sess.run(init_train)
        train_sum_loss = 0.0
        with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=train_gen.get_target_len())) as pbar:
            while True:
                l, _, w = sess.run([lm_loss, lm_train_op, word_ids], feed_dict=feed)
                bsize = len(w)
                train_sum_loss += l * bsize
                if pbar.n % 1000 == 0: pbar.set_description("loss:{loss:.4e}".format(loss=l))
                pbar.update(bsize)
        feed = params.feed({is_training:False})
        sess.run(tf.local_variables_initializer())
        sess.run(init_val)
        val_sum_loss = 0.0
        with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=val_gen.get_target_len())) as pbar:
            while True:
                l, w = sess.run([lm_loss, word_ids], feed_dict=feed)
                bsize = len(w)
                val_sum_loss += l * bsize
                if pbar.n % 1000 == 0: pbar.set_description("loss:{loss:.4e}".format(loss=l))
                pbar.update(bsize)
        print("lm epoch:{}   train_loss:{}   val_loss:{}".format(epoch, train_sum_loss, val_sum_loss))
    return val_sum_loss

def run(params, do_write, name_suffix=None):
    print("run", params.name(name_suffix))
    if SUBMIT:
        saver = tf.train.Saver(max_to_keep=1)
    with contextlib.ExitStack() as stack:
        if do_write:
            train_writer = stack.enter_context(contextlib.closing(tf.summary.FileWriter(os.path.join("tfboard", "train", params.name(name_suffix)))))
            val_writer = stack.enter_context(contextlib.closing(tf.summary.FileWriter(os.path.join("tfboard", "val", params.name(name_suffix)))))
        else:
            train_writer = val_writer = None

        sess.run(tf.global_variables_initializer())

        if params.lm_epoch > 0:
            print("start lm learning")
            do_lm(params.lm_epoch)

        print("start insincere learning")
        val_print_results = []
        best_val_result = {"f_score":-1}

        for epoch in range(params.epoch):
            do_train(epoch, train_writer)
            do_attention(epoch)
            val_result = do_val(epoch, val_writer, num_print_failure=5)
            print_result = "epoch:{} f_score:{} threshold:{} loss:{}".format(epoch, val_result["f_score"], val_result["threshold"], val_result["loss"])
            print(print_result)
            val_print_results.append(print_result)
            if best_val_result["f_score"] < val_result["f_score"]:
                best_val_result = val_result
                if SUBMIT:
                    saver.save(sess, "./model")

            if epoch % HALF_EPOCH == (HALF_EPOCH-1):
                sess.run(half_adam_lr)

        if val_writer is not None:
            val_end_summarize(writer=val_writer, **best_val_result)
    return best_val_result, val_print_results


def do_test(threshold):
    feed = params.feed({is_training:False})
    saver = tf.train.Saver()
    saver.restore(sess, "./model")
    sess.run(tf.local_variables_initializer())
    if params.parameter_average > 0.0:
        average_manager.switch_to_average(sess)
    sess.run(init_test)

    outs = [None for _ in range(test_gen.get_target_len())]
    with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=test_gen.get_target_len())) as pbar:
        while True:
            ps, indices = sess.run([probs, instance_indices], feed_dict=feed)
            for p, i in zip(ps, indices):
                outs[i] = p
            pbar.update(len(ps))
    if params.parameter_average > 0.0:
        average_manager.switch_to_temporal(sess)
    return (threshold, outs)
if SUBMIT:
    if ENSEMBLE == "prob":
        def submit_ensemble(test_results):
            mean_probs = np.mean([result[1] for result in test_results], axis=0)
            mean_threshold = np.mean([result[0] for result in test_results])
            predictions = [int(d) for d in (mean_probs > mean_threshold)]
            submittion = pd.DataFrame({"qid":test_df["qid"].values, "prediction":predictions})
            submittion.to_csv("submission.csv", index=False)
    elif ENSEMBLE == "pred":
        assert NUM_K_FOLD % 2 == 1
        def submit_ensemble(test_results):
            ens_threshold = NUM_K_FOLD // 2 + 1
            predictions = [(np.array(test_probs) >= threshold).astype(np.int32) for threshold, test_probs in test_results]
            predictions = [int(p) for p in (np.sum(predictions, axis=0) >= (ens_threshold - 0.01))]
            submittion = pd.DataFrame({"qid":test_df["qid"].values, "prediction":predictions})
            submittion.to_csv("submission.csv", index=False)
    else:
        raise ValueError("ENSEMBLE:{}".format(ENSEMBLE))


if SUBMIT:
    sess = tf.Session()
    all_val_print_results = []
    test_results = []
    for train_indices, val_indices in cv_splits:
        train_gen.construct(train_indices)
        val_gen.construct(val_indices)
        best_val_result, val_print_results = run(params=params, do_write=True, name_suffix=sys.argv[1] if len(sys.argv) == 2 else "")
        all_val_print_results.append((best_val_result, val_print_results))
        test_result = do_test(best_val_result["threshold"])
        test_results.append(test_result)

    for i, [best_result, results] in enumerate(all_val_print_results):
        print("--------- CV {} / {} ------------".format(i+1, len(all_val_print_results)))
        for result in results:
            print(result)
        print(best_result)

    for best_result, results in all_val_print_results:
        print(best_result)

    submit_ensemble(test_results)

else:
    train_indices, val_indices = cv_splits[0]
    train_gen.construct(train_indices)
    val_gen.construct(val_indices)

    sess = tf.InteractiveSession()
    run(params=params, do_write=True, name_suffix=sys.argv[1] if len(sys.argv) == 2 else "")

