#!/usr/bin/python3
import github # pip3 install PyGithub

REPO_NAME = "pgcool/TF-MTRNN"
DIR_PATH = "data/CoNLL04/"
TARGET_FILES = ["testing_indx_POS_sentence.txt", "testing_indx_ner_sentence_BILOU.txt", "testing_indx_sentence.txt",
                "training_indx_POS_sentence.txt", "training_indx_ner_sentence_BILOU.txt", "training_indx_sentence.txt"]

def main():
    print("donwload...", end="", flush=True)
    #donwload()
    print("ok.", flush=True)

    print("built.")

def donwload():
    g = github.Github()
    repo = g.get_repo(REPO_NAME)
    for fname in TARGET_FILES:
        contents = repo.get_contents(DIR_PATH + fname)
        with open(fname, "wb") as f:
            f.write(contents.decoded_content)

if __name__ == "__main__":
    main()
