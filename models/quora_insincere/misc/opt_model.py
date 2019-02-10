import pdb
pdb.set_trace()
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

SUBMIT = False

def wrapper(trial):
    setting = dict()

    #setting["character_feature"] = trial.suggest_categorical("character_feature", ["NONE", "ConcatCharCNN"])
    int_character_feature = trial.suggest_int("character_feature", 0, 0) # 0: "NONE", 1: "ConcatCharCNN"
    if int_character_feature == 0: setting["character_feature"] = "NONE"
    elif int_character_feature == 1: setting["character_feature"] = "ConcatCharCNN"
    else: raise ValueError("int_character_feature:{}".format(int_character_feature))
    if setting["character_feature"] == "ConcatCharCNN":
        setting["dim_char"] = int(2**trial.suggest_int("ConcatCharCNN_log2_dim_char", 4, 5)) # 2**4=16, 2**5=32
        setting["dim_char_out"] = int(2**trial.suggest_int("ConcatCharCNN_log2_dim_char_out", 4, 6)) # 2**4=16, 2**6=64
        setting["char_cnn_window_size"] = trial.suggest_int("ConcatCharCNN_char_cnn_window_size", 1, 7)

    #setting["batch_size"] = trial.suggest_categorical("batch_size", [256, 512])
    setting["batch_size"] = 512

    log2_cell_size1 = trial.suggest_int("log2_cell_size1", 4, 7)
    setting["cell_size1"] = int(2**log2_cell_size1)
    setting["cell_size2"] = int(2**trial.suggest_int("log2_cell_size2", 4, 7))

    setting["num_head"] = int(2**trial.suggest_int("log2_num_head", 0, 4)) # 2**0=1, 2**4=16
    setting["use_max_pool"] = trial.suggest_categorical("use_max_pool", [False, True])
    setting["use_average_pool"] = trial.suggest_categorical("use_average_pool", [False, True])

    setting["use_hidden_multi"] = trial.suggest_categorical("use_hidden_multi", [False, True])
    if setting["use_hidden_multi"]:
        setting["num_hidden_head"] = int(2**trial.suggest_int("log2_num_hidden_head", 0, 4)) # 2**0=1, 2**4=16
    setting["use_hidden_max_pool"] = trial.suggest_categorical("use_hidden_max_pool", [False, True])
    setting["use_hidden_average_pool"] = trial.suggest_categorical("use_hidden_average_pool", [False, True])

    setting["dim_hidden"] = int(2**trial.suggest_int("log2_dim_hidden", 5, 9)) # 2**5=32, 2**9=512

    setting["keep_prob_input"] = trial.suggest_discrete_uniform("keep_prob_input", 0.4, 1.0, 0.1)
    setting["keep_prob_output"] = trial.suggest_discrete_uniform("keep_prob_output", 0.4, 1.0, 0.1)

    #setting["use_cnn"] = trial.suggest_categorical("use_cnn", [False])

    start_time = time.time()
    print(setting)
    best_val_losses, best_val_loss_results, best_val_loss_epochs, best_val_f_score_losses, best_val_f_score_results, best_val_f_score_epochs = objective(setting)
    end_time = time.time()

    def mean(l):
        return sum(l) / len(l)

    best_loss = mean(best_val_losses)
    best_loss_result = best_val_loss_results
    best_loss_epoch = best_val_loss_epochs
    best_loss_f_score = mean([result["f_score"] for result in best_val_loss_results])
    best_loss_threshold = [result["threshold"] for result in best_val_loss_results]
    trial.set_user_attr("best_loss", best_loss)
    trial.set_user_attr("best_loss_result", best_loss_result)
    trial.set_user_attr("best_loss_epoch", best_loss_epoch)
    trial.set_user_attr("best_loss_f_score", best_loss_f_score)
    trial.set_user_attr("best_loss_threshold", best_loss_threshold)

    best_f_score_loss = mean(best_val_f_score_losses)
    best_f_score_result = best_val_f_score_results
    best_f_score_epoch = best_val_f_score_epochs
    best_f_score = mean([result["f_score"] for result in best_val_f_score_results])
    best_f_score_threshold = [result["threshold"] for result in best_val_f_score_results]
    trial.set_user_attr("best_f_score_loss", best_f_score_loss)
    trial.set_user_attr("best_f_score_result", best_f_score_result)
    trial.set_user_attr("best_f_score_epoch", best_f_score_epoch)
    trial.set_user_attr("best_f_score", best_f_score)
    trial.set_user_attr("best_f_score_threshold", best_f_score_threshold)

    trial.set_user_attr("running_time", end_time - start_time)

    #return best_loss
    return -best_f_score

def objective(setting):
    class Parameter:
        use_pos_tag = False
        use_homebrew = False
        finetune = False
        #batch_size = 256
        batch_size = setting["batch_size"]
        #cell_size1 = 256
        #cell_size1 = 128
        cell_size1 = setting["cell_size1"]
        #cell_size2 = 128
        #cell_size2 = 0
        cell_size2 = setting["cell_size2"]
        #num_head = 8
        num_head = setting["num_head"]
        #dim_hidden = 100
        dim_hidden = setting["dim_hidden"]
        use_skip1 = False
        use_skip2 = False
        label_smoothing = 0
        epoch = 6

        placeholder_keep_prob_input = tf.placeholder(tf.float32, [])
        placeholder_keep_prob_output = tf.placeholder(tf.float32, [])
        def __init__(self):
            #self.keep_prob_input = 0.7
            self.keep_prob_input = setting["keep_prob_input"]
            #self.keep_prob_output = 0.7
            self.keep_prob_output = setting["keep_prob_output"]
        def feed(self, wrap=None):
            out = dict() if wrap is None else dict(wrap)
            out[self.placeholder_keep_prob_input] = self.keep_prob_input
            out[self.placeholder_keep_prob_output] = self.keep_prob_output
            return out
        def name(self, suffix=None):
            stack = []
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
            stack.append("head{num_head}".format(num_head=self.num_head))
            stack.append("hidden{dim_hidden}".format(dim_hidden=self.dim_hidden))
            stack.append("ki{ki}-ko{ko}".format(ki=self.keep_prob_input, ko=self.keep_prob_output))
            if any([self.use_skip1, self.use_skip2]):
                skips = [n for n,f in (("1", self.use_skip1), ("2", self.use_skip2)) if f]
                stack.append("skip" + "-".join(skips))
            if self.label_smoothing > 0.0:
                stack.append("smooth{}".format(self.label_smoothing))

            if suffix is not None and len(suffix) > 0:
                stack.append(suffix)
            return "_".join(stack)

    params = Parameter()


    train_df = pd.read_csv("../input/train.csv")

    preprocessed = np.load("preprocessed.npy").tolist()

    id_to_word = preprocessed["id_to_word"]

    #init_emb = preprocessed["glove_emb"]
    #init_emb = preprocessed["paragram_emb"]
    init_emb = preprocessed["mean_emb"]

    if setting["character_feature"] != "NONE":
        id_to_char = preprocessed["id_to_char"]

    if params.use_homebrew:
        homebrew_init_emb = preprocessed["homebrew_init_emb"]

    if params.use_pos_tag:
        id_to_pos_tag = preprocessed["id_to_pos_tag"]

    all_train_instances = preprocessed["all_train_instances"]
    all_train_labels = [instance["label"] for instance in all_train_instances]
    skfold = StratifiedKFold(5, shuffle=True, random_state=777)
    cv_splits = list(skfold.split(all_train_labels, all_train_labels))
    test_instances = preprocessed["test_instances"]

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
            self.bags = []
            for b in range(0, len(self.target), self.batch_size):
                bag = {}
                for key in keys:
                    value = [instance[key] for instance in self.target[b:b+self.batch_size]]
                    if type(value[0]) in [list, tuple]:
                        if key == "chars":
                            if setting["character_feature"] == "NONE": continue
                            if chars_atom is None:
                                chars_atom = [0] * len(value[0][0])
                            value = pad(value, atom=chars_atom)
                        else:
                            value = pad(value, atom=0)
                    bag[key] = value
                self.bags.append(bag)
            return self
        @property
        def output_shapes(self):
            sample = self.all_instances[0]
            outs = dict()
            for key, value in sample.items():
                if type(value) in [list, tuple]:
                    if key == "chars":
                        if setting["character_feature"] == "NONE": continue
                        outs[key] = tf.TensorShape([None, None, len(value[0])])
                    else:
                        outs[key] = tf.TensorShape([None, None])
                else:
                    outs[key] = tf.TensorShape([None])
            return outs
        @property
        def output_types(self):
            sample = self.all_instances[0]
            outs = dict()
            for key, value in sample.items():
                if key == "chars" and setting["character_feature"] == "NONE": continue

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
            return outs
        def __call__(self):
            order = np.arange(len(self.bags))
            if self.shuffle:np.random.shuffle(order)
            for o in order:
                yield self.bags[o]
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

    word_ids = minibatch.get("word")
    seq_lens = minibatch.get("sequence_length")
    labels = minibatch.get("label")
    instance_indices = minibatch.get("index")


    is_training = tf.placeholder(tf.bool, [])
    def dropout(target, keep_prob):
        return tf.cond(is_training, lambda: tf.nn.dropout(target, keep_prob=keep_prob), lambda: target)


    from attention import MultiHeadReduction
    class Model:
        dim_word_emb = 300
        dim_pos_tag = 30
        num_unit1 = params.cell_size1
        num_unit2 = params.cell_size2
        num_head = params.num_head
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

                #if setting["use_layernorm"]:
                #    import layernorm as LN
                #    self.ln = LN.LayerNormalization(dim=2*self.num_unit1)

                #self.flstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit2)
                #self.blstm2 = tf.contrib.cudnn_rnn.CudnnLSTM(num_layers=1, num_units=self.num_unit2)
                self.flstm2 = tf.keras.layers.CuDNNLSTM(self.num_unit2, return_sequences=True)
                self.blstm2 = tf.keras.layers.CuDNNLSTM(self.num_unit2, return_sequences=True)
                if self.use_skip2:
                    self.w_skip2 = tf.get_variable("w_skip2", [2*self.num_unit1, 2*self.num_unit2])
                self.multi = MultiHeadReduction(num_head=self.num_head, dim_input=2*self.num_unit2)
            else:
                self.multi = MultiHeadReduction(num_head=self.num_head, dim_input=2*self.num_unit1)
            self.fc1 = tf.keras.layers.Dense(self.dim_fc1)
            self.fc_output = tf.keras.layers.Dense(1)

            word_emb = tf.nn.embedding_lookup(params=self.w_word_emb, ids=word_ids) # [batch_size, seq_len, dim_word_emb]

            word_vec = word_emb

            if setting["character_feature"] == "SumBOC":
                self.w_char_emb = tf.get_variable("w_char_emb", [len(id_to_char), init_emb.shape[-1]], initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=True)
                char_ids = minibatch.get("chars") # [batch_size, seq_len, word_len]
                char_emb = tf.nn.embedding_lookup(params=self.w_char_emb, ids=char_ids) # [batch_size, seq_len, word_len, dim_word_emb]
                char_mask = tf.cast(tf.not_equal(char_ids, 0), tf.float32) # [batch_size, seq_len, word_len]
                char_vec = tf.reduce_sum(char_emb * char_mask[:,:,:,tf.newaxis], axis=2) # [batch_size, seq_len, dim_word_emb]
                word_vec = word_vec + char_vec
            elif setting["character_feature"] == "ConcatCharCNN":
                self.w_char_emb = tf.get_variable("w_char_emb", [len(id_to_char), setting["dim_char"]], initializer=tf.truncated_normal_initializer(stddev=0.05), trainable=True)
                self.char_cnn = tf.keras.layers.Conv1D(setting["dim_char_out"], setting["char_cnn_window_size"], padding="same")
                char_ids = minibatch.get("chars") # [batch_size, seq_len, word_len]
                char_emb = tf.nn.embedding_lookup(params=self.w_char_emb, ids=char_ids) # [batch_size, seq_len, word_len, dim_char_emb]
                char_shapes = tf.shape(char_emb)
                char_shapes = [char_shapes[i] for i in range(4)]
                flat_char_emb = tf.reshape(char_emb, [char_shapes[0]*char_shapes[1], char_shapes[2], setting["dim_char"]])
                flat_h_char_cnn = tf.nn.sigmoid(self.char_cnn(flat_char_emb))
                char_mask = tf.cast(tf.not_equal(char_ids, 0), tf.float32) # [batch_size, seq_len, word_len]
                h_char_cnn = tf.reshape(flat_h_char_cnn, char_shapes[:3]+[setting["dim_char_out"]]) * char_mask[:,:,:,tf.newaxis]
                max_h_char_cnn = tf.reduce_max(h_char_cnn, axis=2)
                mean_h_char_cnn = tf.reduce_sum(h_char_cnn, axis=2) / (tf.reduce_sum(char_mask, axis=2)[:,:,tf.newaxis] + 1e-10)
                word_vec = tf.concat([word_vec, max_h_char_cnn, mean_h_char_cnn], axis=2)

            if params.use_homebrew:
                homebrew_word_emb = tf.nn.embedding_lookup(params=self.w_homebrew_word_emb, ids=minibatch["homebrew_word"]) # [batch_size, seq_len, dim_word_emb]
                word_vec = tf.concat([word_emb, homebrew_word_emb], axis=2)

            if params.use_pos_tag:
                pos_tag_emb = tf.nn.embedding_lookup(params=self.w_pos_tag_emb, ids=minibatch["pos"]) # [batch_size, seq_len, dim_word_emb]
                word_vec = tf.concat([word_vec, pos_tag_emb], axis=2)


            word_vec = dropout(word_vec, params.keep_prob_input)

            def reverse_sequence(inputs):
                #return tf.reverse_sequence(inputs, seq_lens, seq_axis=0, batch_axis=1)
                return tf.reverse_sequence(inputs, seq_lens, batch_axis=0, seq_axis=1)
            def bilstm(inputs, flstm, blstm):
                #h_flstm, _ = flstm(inputs)
                #h_blstm, _ = blstm(reverse_sequence(inputs))
                h_flstm = flstm(inputs)
                h_blstm = blstm(reverse_sequence(inputs))
                h_blstm = reverse_sequence(h_blstm)
                return tf.concat([h_flstm, h_blstm], axis=2)

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

                #if setting["use_layernorm"]:
                #    h_bilstm1 = self.ln(h_bilstm1)

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

            input_fc1.append(self.multi(h_bilstm, seq_lens=seq_lens))

            h_bilstm_mask = tf.cast(tf.sequence_mask(seq_lens, tf.shape(word_ids)[1]), tf.float32)[:,:,tf.newaxis]
            masked_h_bilstm = h_bilstm * h_bilstm_mask
            if setting["use_max_pool"]:
                min_value = tf.reduce_min(masked_h_bilstm, axis=1) # [batch_size, dim]
                biased = masked_h_bilstm - min_value[:,tf.newaxis] # all non-masked values are at least 0, but masked values could be over 0
                masked_biased = biased * h_bilstm_mask # all non-masked values are at least 0, while masked values must be 0.
                max_pooled_biased = tf.reduce_max(masked_biased, axis=1)
                max_pooled = max_pooled_biased + min_value
                input_fc1.append(max_pooled)
            if setting["use_average_pool"]:
                average_pooled = tf.reduce_sum(masked_h_bilstm, axis=1) / (tf.cast(seq_lens, tf.float32)+1e-10)[:,tf.newaxis]
                input_fc1.append(average_pooled)

            if setting["use_hidden_multi"]:
                with tf.variable_scope("hidden_multi"):
                    self.hidden_multi = MultiHeadReduction(num_head=setting["num_hidden_head"], dim_input=2*self.num_unit1)
                input_fc1.append(self.hidden_multi(h_bilstm1, seq_lens=seq_lens))
            masked_h_bilstm1 = h_bilstm1 * h_bilstm_mask
            if setting["use_hidden_max_pool"]:
                min_value = tf.reduce_min(masked_h_bilstm1, axis=1) # [batch_size, dim]
                biased = masked_h_bilstm1 - min_value[:,tf.newaxis] # all non-masked values are at least 0, but masked values could be over 0
                masked_biased = biased * h_bilstm_mask # all non-masked values are at least 0, while masked values must be 0.
                max_pooled_biased = tf.reduce_max(masked_biased, axis=1)
                max_pooled = max_pooled_biased + min_value
                input_fc1.append(max_pooled)
            if setting["use_hidden_average_pool"]:
                average_pooled = tf.reduce_sum(masked_h_bilstm1, axis=1) / (tf.cast(seq_lens, tf.float32)+1e-10)[:,tf.newaxis]
                input_fc1.append(average_pooled)

            if len(input_fc1) == 1:
                input_fc1 = input_fc1[0]
            else:
                input_fc1 = tf.concat(input_fc1, axis=-1)

            h_fc1 = tf.nn.tanh(self.fc1(input_fc1))
            h_fc1 = dropout(h_fc1, params.keep_prob_output)
            self.logits = self.fc_output(h_fc1)
    print("build model")
    model = Model()
    logits = model.logits
    loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=labels[:,tf.newaxis], logits=logits, label_smoothing=params.label_smoothing)
    mean_loss, update_mean_loss = tf.metrics.mean(tf.losses.sigmoid_cross_entropy(multi_class_labels=labels[:,tf.newaxis], logits=logits, reduction=tf.losses.Reduction.NONE))
    opt = tf.train.AdamOptimizer()
    train_op = opt.minimize(loss)

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
    def val_end_summarize(writer, threshold, precision, recall, f_score):
        summary = sess.run(merged_val_end, feed_dict={placeholder_best_threshold:threshold,
                                                      placeholder_best_balanced_precision:precision,
                                                      placeholder_best_balanced_recall:recall,
                                                      placeholder_best_balanced_f_score:f_score})
        writer.add_summary(summary, 0)
        writer.flush()


    def do_val(epoch, writer, num_print_failure=0):
        feed = params.feed({is_training:False})
        sess.run(tf.local_variables_initializer())
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
        val_loss = float(sess.run(mean_loss, feed_dict=feed))
        print("val  ", acc, prec, rec, f, val_loss)

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
        outs = {"f_score":float(best_f_score), "threshold":float(best_threshold), "precision":float(best_precision), "recall":float(best_recall)}
        if writer is not None:
            val_epoch_summarize(writer=writer, epoch=epoch, **outs)

        if num_print_failure > 0:
            false_negative = [(p,g,idx) for p,g,idx in zip(val_all_probs, val_all_labels, val_all_indices) if g==1 and g != int(p>=best_threshold)]
            false_positive = [(p,g,idx) for p,g,idx in zip(val_all_probs, val_all_labels, val_all_indices) if g==0 and g != int(p>=best_threshold)]
            np.random.shuffle(false_negative)
            np.random.shuffle(false_positive)
            for target in [false_negative, false_positive]:
                for p,g,idx in target[:num_print_failure]:
                    print(train_df.target[idx], g, p, train_df.question_text[idx])

        return outs, val_loss
    def do_train(epoch, writer):
        feed = params.feed({is_training:True})
        sess.run(tf.local_variables_initializer())
        sess.run(init_train)

        with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=train_gen.get_target_len())) as pbar:
            while True:
                p, _, l, acc, rec, prec, f = sess.run([probs, train_op, update_mean_loss, update_accuracy, update_recall, update_precision, f_score], feed_dict=feed)
                if pbar.n % 1000 == 0: pbar.set_description("loss:{loss:.4e} acc:{accuracy:.4f} prec:{precision:.4f} rec:{recall:.4f} f:{f_score:.4f}".format(loss=l, accuracy=acc, precision=prec, recall=rec, f_score=f))
                pbar.update(len(p))
        if writer is not None:
            train_epoch_summarize(writer, epoch)


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
            true_val_results = []
            val_results = []
            best_val_result = {"f_score":-1}
            val_losses = []

            for epoch in range(params.epoch):
                do_train(epoch, train_writer)
                val_result, val_loss = do_val(epoch, val_writer, num_print_failure=5)
                print_result = "epoch:{} f_score:{} threshold:{} loss:{}".format(epoch, val_result["f_score"], val_result["threshold"], val_loss)
                print(print_result)
                val_results.append(print_result)
                true_val_results.append(val_result)
                if best_val_result["f_score"] < val_result["f_score"]:
                    best_val_result = val_result
                    if SUBMIT:
                        saver.save(sess, "./model")
                val_losses.append(val_loss)
            if val_writer is not None:
                val_end_summarize(writer=val_writer, **best_val_result)
        return best_val_result, true_val_results, val_losses


    def do_test(threshold):
        feed = params.feed({is_training:False})
        saver = tf.train.Saver()
        saver.restore(sess, "./model")
        sess.run(tf.local_variables_initializer())
        sess.run(init_test)

        outs = [None for _ in range(test_gen.get_target_len())]
        with contextlib.suppress(tf.errors.OutOfRangeError), contextlib.closing(tqdm.tqdm(total=test_gen.get_target_len())) as pbar:
            while True:
                ps, indices = sess.run([probs, instance_indices], feed_dict=feed)
                for p, i in zip(ps, indices):
                    outs[i] = p
                pbar.update(len(ps))
        return (threshold, outs)
    def submit_ensemble(test_results):
        mean_probs = np.mean([result[1] for result in test_results], axis=0)
        mean_threshold = np.mean([result[0] for result in test_results])
        predictions = [int(d) for d in (mean_probs > mean_threshold)]
        submittion = pd.DataFrame({"qid":test_df["qid"].values, "prediction":predictions})
        submittion.to_csv("submission.csv", index=False)

    #with tf.Session() as sess:
    #    all_val_results = []
    #    test_results = []
    #    for train_indices, val_indices in cv_splits:
    #        train_gen.construct(train_indices)
    #        val_gen.construct(val_indices)
    #        best_val_result, val_results = run(params=params, do_write=True, name_suffix=sys.argv[1] if len(sys.argv) == 2 else "")
    #        all_val_results.append((best_val_result, val_results))
    #        test_result = do_test(overall_best_results["threshold"])
    #        test_results.append(test_result)

    sess = tf.InteractiveSession()

    best_val_losses = []
    best_val_loss_results = []
    best_val_loss_epochs = []
    best_val_f_score_losses = []
    best_val_f_score_results = []
    best_val_f_score_epochs = []
    for cv_split in cv_splits:
        train_indices, val_indices = cv_split
        train_gen.construct(train_indices)
        val_gen.construct(val_indices)

        best_val_result, val_results, val_losses = run(params=params, do_write=False, name_suffix=sys.argv[1] if len(sys.argv) == 2 else "")

        best_val_loss = np.inf
        best_val_f_score = -np.inf
        for epoch, [val_loss, val_result] in enumerate(zip(val_losses, val_results)):
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                best_val_loss_epoch = epoch
                best_val_loss_result = val_result
            if best_val_f_score < val_result["f_score"]:
                best_val_f_score = val_result["f_score"]
                best_val_f_score_loss = val_loss
                best_val_f_score_epoch = epoch
                best_val_f_score_result = val_result
        assert min(val_losses) == best_val_loss
        assert max(result["f_score"] for result in val_results) == best_val_f_score

        best_val_losses.append(best_val_loss)
        best_val_loss_results.append(best_val_loss_result)
        best_val_loss_epochs.append(best_val_loss_epoch)
        best_val_f_score_losses.append(best_val_f_score_loss)
        best_val_f_score_results.append(best_val_f_score_result)
        best_val_f_score_epochs.append(best_val_f_score_epoch)

    return best_val_losses, best_val_loss_results, best_val_loss_epochs, best_val_f_score_losses, best_val_f_score_results, best_val_f_score_epochs

import optuna
OPTUNA_STUDY_NAME = "model"
OPTUNA_STORAGE = "sqlite:///opt.db"

study = optuna.Study(study_name=OPTUNA_STUDY_NAME, storage=OPTUNA_STORAGE)
study.optimize(wrapper, n_trials=1)

