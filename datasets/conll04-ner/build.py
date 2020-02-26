#!/usr/bin/python3

# http://www.aclweb.org/anthology/C16-1239
# https://github.com/xingdi-eric-yuan/recurrent-net-lstm/tree/master/dataset/CoNLL04

import os
import json
import github # pip3 install PyGithub

import argparse
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dest_dir", type=str, default="./conll04")
    args = parser.parse_args()
    return args
args = parse_args()

REPO_NAME = "pgcool/TF-MTRNN"
REPO_DIR_PATH = "data/CoNLL04/"
TRAIN_SENTENCE_FILE = "training_indx_sentence.txt"
TRAIN_POS_FILE = "training_indx_POS_sentence.txt"
TRAIN_NER_FILE = "training_indx_ner_sentence_BILOU.txt"
TEST_SENTENCE_FILE = "testing_indx_sentence.txt"
TEST_POS_FILE = "testing_indx_POS_sentence.txt"
TEST_NER_FILE = "testing_indx_ner_sentence_BILOU.txt"
TARGET_FILES = [TRAIN_SENTENCE_FILE, TRAIN_POS_FILE, TRAIN_NER_FILE,
                TEST_SENTENCE_FILE, TEST_POS_FILE, TEST_NER_FILE]
DOWNLOAD_DIR = os.path.join(args.dest_dir, "download")
TRAIN_OUTPUT_FILE = os.path.join(args.dest_dir, "train.json")
TEST_OUTPUT_FILE = os.path.join(args.dest_dir, "test.json")

def main():
    print("downloading...", end="", flush=True)
    download()
    print("ok.", flush=True)

    print("loading training files...", end="", flush=True)
    train = load(sent_key=TRAIN_SENTENCE_FILE, pos_key=TRAIN_POS_FILE, ner_key=TRAIN_NER_FILE, do_unnormalize=True)
    print("ok.", flush=True)

    print("loading test files...", end="", flush=True)
    test = load(sent_key=TEST_SENTENCE_FILE, pos_key=TEST_POS_FILE, ner_key=TEST_NER_FILE, do_unnormalize=True)
    print("ok.", flush=True)

    print("dumping...", end="", flush=True)
    with open(TRAIN_OUTPUT_FILE, "w") as f:
        json.dump(train, f)
    with open(TEST_OUTPUT_FILE, "w") as f:
        json.dump(test, f)
    print("ok.", flush=True)

    print(f"succeeded in constructing '{TRAIN_OUTPUT_FILE}' and '{TEST_OUTPUT_FILE}'.")

def download():
    g = github.Github()
    repo = g.get_repo(REPO_NAME)
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    for fname in TARGET_FILES:
        contents = repo.get_contents(os.path.join(REPO_DIR_PATH, fname))
        with open(os.path.join(DOWNLOAD_DIR, fname), "wb") as f:
            f.write(contents.decoded_content)

def load(sent_key, pos_key, ner_key, do_unnormalize):
    sent_path, pos_path, ner_path = map(lambda x:os.path.join(DOWNLOAD_DIR, x), [sent_key, pos_key, ner_key])
    data = {}
    with open(sent_path) as f:
        for line in f:
            sent_id, *words = line.strip().split()
            sequence_length = len(words)
            assert sequence_length > 0
            assert sent_id[-1] == ":"
            sent_id = sent_id[:-1]
            assert sent_id not in data

            if do_unnormalize: words = unnormalize(words)

            data[sent_id] = {"words": words, "sequence_length":sequence_length}

    with open(pos_path) as f:
        for line in f:
            sent_id, colon, *poss = line.strip().split()
            assert len(poss) == data[sent_id]["sequence_length"]
            assert colon == ":"
            assert sent_id in data
            assert "POSs" not in data[sent_id]

            data[sent_id]["POSs"] = poss

    with open(ner_path) as f:
        for line in f:
            sent_id, colon, *entities = line.strip().split()
            assert len(entities) == data[sent_id]["sequence_length"]
            assert colon == ":"
            assert sent_id in data
            assert "entities" not in data[sent_id]

            data[sent_id]["entities"] = entities

    list_data = []
    for sent_id, values in data.items():
        copied = dict(values)
        copied["sentence_id"] = sent_id
        list_data.append(copied)

    return list_data

def unnormalize(words, do_comma=True, do_rb=True, do_cb=True, do_sb=True):
    unnormalized = []
    for word in words:
        if do_comma and word == "COMMA": word = ","
        if do_rb:
            if word == "-LRB-": word = "("
            if word == "-RRB-": word = ")"
        if do_cb:
            if word == "-LCB-": word = "{"
            if word == "-RCB-": word = "}"
        if do_sb:
            if word == "-LSB-": word = "["
            if word == "-RSB-": word = "]"
        unnormalized.append(word)
    return unnormalized

if __name__ == "__main__":
    main()
