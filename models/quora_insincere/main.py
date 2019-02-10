import sys, os
import math
import collections
import re
import multiprocessing
import time
import contextlib
import json
import tqdm
def ctqdm(*args, **kwargs): return contextlib.closing(tqdm.tqdm(*args, **kwargs))

import nltk
import numpy as np
import tensorflow as tf
import pandas as pd
import sentencepiece as spm

GLOVE_PATH = "../input/embeddings/glove.840B.300d/glove.840B.300d.txt"
PARAGRAM_PATH = "../input/embeddings/paragram_300_sl999/paragram_300_sl999.txt"

MAX_SEQ_LEN = 400

USE_CHARACTER = False
USE_REPLACE_TOKEN = False
USE_POS = False
USE_HOMEBREW = False
USE_SENTENCE_PIECE = False
SAVE = True

assert not USE_REPLACE_TOKEN or not USE_SENTENCE_PIECE

# preload
if USE_POS:
    nltk.pos_tag(["this", "is", "test"])
    nltk.stem.WordNetLemmatizer().lemmatize("test")

#----------------------------------------------------------------------------
print("load csv", end="...", flush=True)
train_df = pd.read_csv("../input/train.csv")
test_df = pd.read_csv("../input/test.csv")
print("done.")
#/---------------------------------------------------------------------------


#----------------------------------------------------------------------------
NUM_KEYS = [str(i) for i in range(10)] + ["="]
RE_SINGLE_NUM = re.compile("[0-9]")
MATH_TOKEN = "MATHTOKEN"
FWORD_TOKEN = "FWORDTOKEN"
ONLY_STAR_TOKEN = "ONLYSTARTOKEN"

class SuffixCounter:
    def __init__(self):
        self.counter = 0
    def new(self):
        suffix = chr(ord("A")+self.counter)
        self.counter += 1
        return suffix

def preprocess_sent(sent):
    suffix = SuffixCounter()
    token_map = dict()

    # irreversible transformation
    sent = sent.replace("\xa0", " ")
    sent = sent.replace("\u200b", " ")
    sent = sent.strip()

    # [0-9] => 9
    #sent = RE_SINGLE_NUM.sub("9", sent)

    # [math]-[math] => MATH

    # split
    words = sent.split()

    # ?-split
    new_words = []
    for word in words:
        if "?" not in word:
            new_words.append(word)
        else:
            new_words.extend([subword for subword in re.split(r"(\?)", word) if len(subword) > 0])
            """
            q_splits = word.split("?")
            new_words.append(q_splits[0])
            for q_split in q_splits[1:]:
                new_words.append("?")
                new_words.append(q_split)
            """
    words = new_words

    # fword
    new_words = []
    for word in words:
        if "*" in word:
            if len(set(word)) == 1: # only "*"
                after = ONLY_STAR_TOKEN + suffix.new()
                new_words.append(after)
                token_map[after] = word
            else:
                if any(num_key in word for num_key in NUM_KEYS):
                    after = MATH_TOKEN + suffix.new()
                    new_words.append(after)
                    token_map[after] = word
                elif "*" not in word[1:-1]:
                    if word[0] == "*":
                        after = ONLY_STAR_TOKEN + suffix.new()
                        new_words.append(after)
                        token_map[after] = "*"
                        word = word[1:]
                    if word[-1] == "*":
                        new_words.append(word[:-1])

                        after = ONLY_STAR_TOKEN + suffix.new()
                        new_words.append(after)
                        token_map[after] = "*"
                    else:
                        new_words.append(word)
                else:
                    after = FWORD_TOKEN + suffix.new()
                    new_words.append(after)
                    token_map[after] = word
        else: # "*" not in word
            new_words.append(word)

    preprocessed_sent = " ".join(new_words)
    if suffix.counter > 20: print(sent)
    return preprocessed_sent, token_map

def postprocess_sent(tokenized, token_map):
    suffix = SuffixCounter()
    suffix.counter = len(token_map)

    # url置換
    # "://" ".com" ".dll" ".info" ".exe"

    # - split
    # 全部知っていたらsplit
    # 数字のみから構成されるならそのまま
    # 両脇が数字なら特殊化

    # 数字置換
    # 0-9+ -> 9
    # A-Z -> X
    # a-z -> x

    # 's
    # n't

    # 簡易版
    new_tokenized = []
    for token in tokenized:
        if "-" not in token:
            new_tokenized.append(token)
        else:
            if token == "-":
                new_tokenized.append(token)
            elif "9" not in token:
                new_tokenized.extend([subword for subword in re.split("(-)", token) if len(subword) > 0])
                """
                for word in token.split("-"):
                    if len(word) == 0: continue
                    new_tokenized.append(word)
                """
            else:
                new_tokenized.append(token)
    tokenized = new_tokenized
    new_tokenized = []
    for token in tokenized:
        if token[-2:].lower() == "'s":
            if len(token) > 2:
                new_tokenized.append(token[:-2])
            new_tokenized.append("'s")
        elif token[-3:].lower() == "n't":
            if len(token) > 3:
                new_tokenized.append(token[:-3])
            new_tokenized.append("n't")
        elif token[-2:].lower() == "'t":
            if len(token) > 2:
                new_tokenized.append(token[:-2])
            new_tokenized.append("'t")
        else:
            if len(set(token)) == 1:
                new_tokenized.append(token)
            else:
                new_tokenized.append(token.replace("'", ""))
    tokenized = new_tokenized
    new_tokenized = []
    for token in tokenized:
        if "9" not in token:
            new_tokenized.append(token)
        else:
            new_tokenized.append(re.sub("a-z", "x", re.sub("A-Z", "X", token)))
    tokenized = new_tokenized
    if USE_REPLACE_TOKEN:
        for after_token in token_map:
            idx = next(filter(lambda x: after_token == x[1], enumerate(tokenized)))[0]
            tokenized[idx] = after_token[:-1]
    else:
        for after_token, before_token in token_map.items():
            idx = next(filter(lambda x: after_token == x[1], enumerate(tokenized)))[0]
            tokenized[idx] = before_token

    if MAX_SEQ_LEN > 0:
        tokenized = tokenized[:MAX_SEQ_LEN]

    if not USE_POS:
        return tokenized, token_map

    pos_tags = []
    lemmatized = []

    word_and_pos_tags = nltk.pos_tag(tokenized)
    _, pos_tags = zip(*word_and_pos_tags)
    lemmatizer = nltk.stem.WordNetLemmatizer()
    wordnet_pos = lambda e: ('a' if e[0].lower() == 'j' else e[0].lower()) if e[0].lower() in ['n', 'r', 'v'] else 'n'
    lemmatize = lambda x: lemmatizer.lemmatize(x[0].lower(), wordnet_pos(x[1]))
    lemmatized = [lemmatize(w) for w in word_and_pos_tags]

    return tokenized, pos_tags, lemmatized, token_map



def tokenize(sent):
    sent, token_map = preprocess_sent(sent)
    tokenizer = nltk.tokenize.TweetTokenizer()
    tokenized = tokenizer.tokenize(sent)

    #tokenized, pos_tags, lemmatized, token_map = postprocess_sent(tokenized, token_map)
    #return tokenized, pos_tags, lemmatized, token_map
    return postprocess_sent(tokenized, token_map)

print("tokenize", end="...", flush=True)
s = time.time()
with multiprocessing.Pool(8) as pool:
    if not USE_POS:
        all_train_sents, train_token_map = zip(*pool.map(tokenize, train_df.question_text))
    else:
        all_train_sents, all_train_pos_tags, all_train_lemmas, train_token_map = zip(*pool.map(tokenize, train_df.question_text))
with multiprocessing.Pool(8) as pool:
    if not USE_POS:
        test_sents, test_token_map = zip(*pool.map(tokenize, test_df.question_text))
    else:
        test_sents, test_pos_tags, test_lemmas, test_token_map = zip(*pool.map(tokenize, test_df.question_text))
print("done.", time.time() - s)
#/---------------------------------------------------------------------------

print("build vocab", end="...", flush=True)
train_vocab_counter = collections.Counter([word for sent in all_train_sents for word in sent])
test_only_vocab = {word for sent in test_sents for word in sent} - set(train_vocab_counter)
word_to_id = {word:id_+1 for id_,word in enumerate(sorted(set(train_vocab_counter) | test_only_vocab))}
word_to_id["$$UNK$$"] = 0
id_to_word = [word for word,id_ in sorted(word_to_id.items(), key=lambda x:x[1])]
print("done.", flush=True)

def load_embedding(fname, word_to_id, train_vocab_counter, logarithm=True, do_gc=False, paragram=False):
    if paragram:
        pretrainable_vocab = {word.lower() for word in word_to_id}
    else:
        pretrainable_vocab = set(word_to_id)
    pretrainable_vocab.update(["*", "fuck", "shit"])

    word_to_vec = dict()
    with open(fname, encoding="latin-1" if paragram else "utf-8") as f:
        for line in f:
            line = line.rstrip()
            if len(line) == 0: continue
            word, *vec = line.split(" ")
            if word in pretrainable_vocab:
                vec = np.array([float(v) for v in vec], dtype=np.float32)
                word_to_vec[word] = vec

    pretrained_train_words = [word for word in train_vocab_counter if (word.lower() if paragram else word) in word_to_vec]
    distribution = [train_vocab_counter[word] for word in pretrained_train_words]
    if logarithm:
        distribution = np.log(distribution)
    unk_vec = np.average([word_to_vec[word.lower() if paragram else word] for word in pretrained_train_words], axis=0, weights=distribution)

    outs = np.tile(unk_vec[np.newaxis,:], [len(word_to_id), 1])
    for word, id_ in word_to_id.items():
        if paragram:
            word = word.lower()
        if word in word_to_vec:
            outs[id_] = word_to_vec[word]

    if USE_REPLACE_TOKEN:
        if ONLY_STAR_TOKEN in word_to_id:
            outs[word_to_id[ONLY_STAR_TOKEN]] = word_to_vec["*"]
        if FWORD_TOKEN in word_to_id:
            outs[word_to_id[FWORD_TOKEN]] = np.mean([word_to_vec["fuck"], word_to_vec["shit"]], axis=0)
        if MATH_TOKEN in word_to_id:
            outs[word_to_id[MATH_TOKEN]] = word_to_vec["*"]

    oov = pretrainable_vocab - set(word_to_vec)
    if do_gc:
        del pretrainable_vocab, word_to_vec, pretrained_train_words, distribution, unk_vec
        import gc
        print("gc:", gc.collect())
    return outs, oov
print("load glove", end="...", flush=True)
s = time.time()
glove_emb, glove_oov = load_embedding(GLOVE_PATH, word_to_id, train_vocab_counter, logarithm=True)
e = time.time()
print("done.", e-s, flush=True)
print("load paragram", end="...", flush=True)
s = time.time()
paragram_emb, paragram_oov = load_embedding(PARAGRAM_PATH, word_to_id, train_vocab_counter, logarithm=True, paragram=True)
e = time.time()
print("done.", e-s, flush=True)


# character
if USE_CHARACTER:
    train_char_counter = collections.Counter()
    for word, count in train_vocab_counter.items():
        sub_counter = collections.Counter(word * count)
        train_char_counter.update(sub_counter)

    MIN_CHAR_FREQUENCY = 1000
    char_to_id = {char:i+3 for i,char in enumerate(sorted([char for char,count in train_char_counter.items() if count >= MIN_CHAR_FREQUENCY]))}
    char_to_id["$$PAD$$"] = 0
    char_to_id["$$CENTER$$"] = 1
    char_to_id["$$UNK$$"] = 2
    unk_char_id = char_to_id["$$UNK$$"]
    id_to_char = [char for char,id_ in sorted(char_to_id.items(), key=lambda x:x[1])]

    MAX_WORD_LEN = 13
    def func_word_to_chars(word):
        if len(word) <= MAX_WORD_LEN:
            return ([char_to_id.get(char, unk_char_id) for char in word] + [0] * (MAX_WORD_LEN-len(word)), len(word))
        else:
            center = [char_to_id["$$CENTER$$"]]
            if MAX_WORD_LEN % 2 == 0:
                l = MAX_WORD_LEN // 2 - 1
                center = center * 2
            else:
                l = MAX_WORD_LEN // 2
            return ([char_to_id.get(char, unk_char_id) for char in word[:l]] + center + [char_to_id.get(char, unk_char_id) for char in word[-l:]], MAX_WORD_LEN)
    word_to_chars = {word:func_word_to_chars(word) for word in sorted(set(train_vocab_counter) | test_only_vocab)}


# homebrew
if USE_HOMEBREW:
    print("homebrew")
    dim_homebrew = 50 # 66.5% (default) 1epoch目で65.91、2epoch目で66.53
    glove_window = 15
    glove_iter = 15
    glove_min = 5
    glove_lower = False

    #dim_homebrew = 300 # 66.5% 学習が早い。66.32->66.49。word-simはぱっと見変わらないがロスは小さい
    #dim_homebrew, glove_iter = 300, 50 # 66.4% 66.40->66.16
    #dim_homebrew = 150 # 66.6% 66.04->66.58
    #glove_window = 7 # 66.4% 65.51->66.41。word-simは強く関連してそうなものだけ残って変な単語が減る。
    #glove_window = 11 # 66.4% 65.54->66.35
    #glove_iter = 50 # 66.4% 65.69->66.36 it15とword-simはスコア含めほぼ変わらないように見える。ロスは1割ほど落ちた。(iter15=0.040320, iter50=0.036944)
    #glove_min = 50 # 66.2% 悪い。ねばる。64.98->66.20->66.21。word-simはぱっと見変わらない。よく見るとレアワードでちゃんと変わってるかも。
    #glove_lower = True # 66.7% 65.61->66.68
    #dim_homebrew, glove_lower = 300, True # 66.18->66.69

    homebrew_word_to_id = {"<unk>":0}
    homebrew_id_to_word = ["<unk>"]
    homebrew_new_id = 1
    homebrew_init_emb = []
    with open("../homebrew/glove-homebrew{}.{}d.win{}-it{}-min{}.txt".format((".lower" if glove_lower else ""), dim_homebrew, glove_window, glove_iter, glove_min)) as f:
        for line in f:
            line = line.strip()
            if len(line) == 0: continue
            word, *vec = line.split(" ")
            assert len(vec) == dim_homebrew
            vec = np.array([float(v) for v in vec], dtype=np.float32)
            if word == "<unk>":
                homebrew_init_emb = [vec] + homebrew_init_emb
                continue
            homebrew_word_to_id[word] = homebrew_new_id
            homebrew_new_id += 1
            homebrew_id_to_word.append(word)
            homebrew_init_emb.append(vec)
    homebrew_init_emb = np.stack(homebrew_init_emb, axis=0)



if USE_POS:
    pos_tag_set = {pos_tag for sents in [all_train_pos_tags, test_pos_tags] for sent in sents for pos_tag in sent}
    #id_to_pos_tag = ["$$UNK$$"] + list(pos_tag_set)
    id_to_pos_tag = list(pos_tag_set)
    pos_tag_to_id = {t:i for i,t in enumerate(id_to_pos_tag)}
    all_train_pos_tags = [[pos_tag_to_id[pos_tag] for pos_tag in sent] for sent in all_train_pos_tags]
    test_pos_tags = [[pos_tag_to_id[pos_tag] for pos_tag in sent] for sent in test_pos_tags]

if USE_SENTENCE_PIECE:
    with open("sentences.txt", "w") as f:
        for sents in [all_train_sents, test_sents]:
            for words in sents:
                print(" ".join(words), file=f)

    SP_VOCAB_SIZE = 2048
    spm.SentencePieceTrainer.Train('--input=sentences.txt --model_prefix=sp{vocab} --vocab_size={vocab} --character_coverage=0.9995'.format(vocab=SP_VOCAB_SIZE))

    sp = spm.SentencePieceProcessor()
    sp.Load('sp{}.model'.format(SP_VOCAB_SIZE))
    with ctqdm(sorted(set(train_vocab_counter) | test_only_vocab), desc="build sp map") as vocab:
        word_to_sp = {word:sp.EncodeAsIds(word) for word in vocab}


def to_instance(idx, sent, label, pos=None, lemma=None):
    outs = dict()
    outs["word"] = [word_to_id.get(word, 0) for word in sent]
    outs["sequence_length"] = len(sent)
    outs["label"] = label
    outs["index"] = idx
    if USE_HOMEBREW:
        outs["homebrew_word"] = [homebrew_word_to_id.get(word.lower() if glove_lower else word, 0) for word in sent]
    if USE_POS:
        outs["pos"] = pos
        outs["lemma"] = lemma
    if USE_CHARACTER:
        chars, char_lens = zip(*[word_to_chars[word] for word in sent])
        outs["chars"] = chars
        outs["char_lens"] = char_lens
    if USE_SENTENCE_PIECE:
        outs["sp"] = [word_to_sp[word] for word in sent]
    return outs
if not USE_POS:
    all_train_instances = [to_instance(idx, sent, label) for idx, [sent,label] in enumerate(zip(all_train_sents, train_df.target))]
    test_instances = [to_instance(idx, sent, 0) for idx, sent in enumerate(test_sents)]
else:
    all_train_instances = [to_instance(idx, sent, label, pos, lemma) for idx, [sent,label,pos,lemma] in enumerate(zip(all_train_sents, train_df.target, all_train_pos_tags, all_train_lemmas))]
    test_instances = [to_instance(idx, sent, 0, pos, lemma) for idx, [sent,pos,lemma] in enumerate(zip(test_sents, test_pos_tags, test_lemmas))]
all_train_instances = np.array(all_train_instances)
test_instances = np.array(test_instances)


save = {"all_train_instances":all_train_instances,
        "test_instances":test_instances,
        "id_to_word":id_to_word,
        "glove_emb":glove_emb,
        "glove_oov":glove_oov,
        "paragram_emb":paragram_emb,
        "mean_emb":np.mean([glove_emb, paragram_emb], axis=0),
}
if USE_HOMEBREW:
    save["homebrew_init_emb"] = homebrew_init_emb
if USE_POS:
    save["id_to_pos_tag"] = id_to_pos_tag
if USE_CHARACTER:
    save["id_to_char"] = id_to_char
if USE_SENTENCE_PIECE:
    save["sp_bos_eos"] = [sp.bos_id(), sp.eos_id()]


if SAVE:
    np.save("preprocessed", np.array(save))
exit(0)

