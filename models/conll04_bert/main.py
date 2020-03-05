import sys, os
import json
import argparse
import tqdm
from contextlib import ExitStack as NullContext

import numpy as np
import torch
import transformers

import baseline.torch
from baseline.utils import BasicDictionary
from baseline.torch.utils import get_transformers_output_dim, flatten
from baseline.utils import SubwordOffsetTokenizer, bilou_span_decode, compute_precision_recall_f

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda:0"])
    parser.add_argument("--train_json", type=str, required=True, help="path to train.json")
    parser.add_argument("--test_json", type=str, required=True, help="path to test.json")

    parser.add_argument("--bert_class", type=str, default="bert", choices=["bert", "roberta"])
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate","--lr", type=float, default=2e-5)

    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    return args
args = parse_args()
bert_class_to_transformers_key = {
    "bert": "bert-base-uncased",
    "roberta": "roberta-base",
}
transformers_key = bert_class_to_transformers_key[args.bert_class]
device = torch.device(args.device)


tokenizer = transformers.AutoTokenizer.from_pretrained(transformers_key)
subword_offset_tokenizer = SubwordOffsetTokenizer(tokenizer)
pos_dic = BasicDictionary("$$UNK$$")
ent_dic = BasicDictionary("$$UNK$$")
ent_cat_dic = BasicDictionary("$$UNK$$")
bilou_dic = BasicDictionary("$$UNK$$")

def preprocess(instance):
    out = dict(instance)

    out.update(subword_offset_tokenizer.encode(instance["words"])) # add 'input_ids' and 'subword_offsets'.
    out["POSs"] = pos_dic.add(instance["POSs"])
    out["bilous"] = bilou_dic.add([e.split("-")[0] for e in instance["entities"]])
    out["entity_categories"] = ent_cat_dic.add([e.split("-")[-1] for e in instance["entities"]])
    out["entities"] = ent_dic.add(instance["entities"])
    return out
def load_json(fname):
    with open(fname) as f:
        loaded = json.load(f)
    loaded = [preprocess(instance) for instance in tqdm.tqdm(loaded)]
    return loaded
selectors = [
    {"name":"input_ids", "dtype":torch.long, "device":device, "padding":True, "padding_value":0, "padding_mask":True},
    {"name":"word_sequence_length", "origin":"sequence_length", "dtype":torch.long, "device":device},
    {"name":"POSs", "dtype":torch.long, "device":device, "padding":True, "padding_value":-1, "padding_mask":True},
    {"name":"entities", "dtype":torch.long, "device":device, "padding":True, "padding_value":-1, "padding_mask":True},
    {"name":"bilous", "dtype":torch.long, "device":device, "padding":True, "padding_value":-1, "padding_mask":True},
    {"name":"entity_categories", "dtype":torch.long, "device":device, "padding":True, "padding_value":-1, "padding_mask":True},
    {"name":"subword_offset_matrix", "mapping":subword_offset_tokenizer.get_offset_matrix, "dtype":torch.float, "device":device, "padding":True, "padding_value":0.0},
]
train_dataset = baseline.torch.utils.SelectiveDataset(load_json(args.train_json), selectors)
test_dataset = baseline.torch.utils.SelectiveDataset(load_json(args.test_json), selectors)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.transformers_key = transformers_key
        self.bert = transformers.AutoModel.from_pretrained(self.transformers_key)
        self._bert_dim = get_transformers_output_dim(self.bert)
        self.fc1 = torch.nn.Linear(self._bert_dim, args.hidden_dim)
        self.fc_fullEntity_output = torch.nn.Linear(args.hidden_dim, len(ent_dic))

    def forward(self, inputs):
        h, pooled = self.bert(input_ids=inputs["input_ids"], attention_mask=inputs["input_ids_mask"]) # [batch_size, subword_seq_len, bert_dim], [*]
        h = self.fc1(h).tanh() # [batch_size, subword_seq_len, hidden_dim]
        logits = self.fc_fullEntity_output(h) # [batch_size, subword_seq_len, num_entities]
        logits = (logits[:,:,None] * inputs["subword_offset_matrix"][:,:,:,None]).sum(1) # [batch_size, word_seq_len, num_entities]
        return logits
model = Model()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)

ce_loss = torch.nn.CrossEntropyLoss(ignore_index=-1)
def loss_func(logits, minibatch):
    labels = minibatch["entities"]
    return ce_loss(flatten(logits, [0,1]), flatten(labels, [0,1]))

def compute_metrics(logits, minibatch, prev_metrics=None):
    labels = minibatch["entities"]
    flat_labels = flatten(labels, [0,1])
    preds = logits.argmax(-1)
    flat_preds = flatten(preds, [0,1])
    isnot_padded = (flat_labels != -1)
    num_token = isnot_padded.sum().item()
    is_correct = (flat_labels == flat_preds) * isnot_padded
    num_token_correct = is_correct.sum().item()

    num_positive = 0
    num_positive_pred = 0
    num_positive_correct = 0
    for seq_pred, seq_label in zip(preds.tolist(), labels.tolist()):
        seq_pred, seq_label = map(ent_dic.decode, [seq_pred, seq_label])
        pred_spans, label_spans = map(lambda x: set(bilou_span_decode(x, [ent_dic.unk_symbol])), [seq_pred, seq_label])
        num_positive += len(label_spans)
        num_positive_pred += len(pred_spans)
        num_positive_correct += len(label_spans & pred_spans)

    out = {"num_token": num_token, "num_token_correct":num_token_correct, "num_positive":num_positive, "num_positive_pred":num_positive_pred, "num_positive_correct":num_positive_correct}
    if prev_metrics is not None:
        out = {key:out[key]+prev_metrics[key] for key in out.keys()}
    out["token_accuracy"] = out["num_token_correct"] / max(1, out["num_token"])
    out.update(compute_precision_recall_f(true_positive=out["num_positive_correct"], real_positive=out["num_positive"], pred_positive=out["num_positive_pred"]))
    return out

def do_iteration(is_training):
    if is_training:
        model.train()
        dataset = train_dataset
    else:
        model.eval()
        dataset = test_dataset
    total_loss = 0.0
    metrics = None
    with NullContext() if is_training else torch.no_grad():
        for minibatch in tqdm.tqdm(dataset.dataloader(args.batch_size, True)):
            if is_training:
                model.zero_grad()
            logits = model(minibatch)
            loss = loss_func(logits, minibatch)
            if is_training:
                loss.backward()
                optimizer.step()
            total_loss += loss.item()
            metrics = compute_metrics(logits, minibatch, prev_metrics=metrics)
    return total_loss, metrics


best_test_metrics = {"f-score":-1.0}
for epoch in range(1,args.epoch+1):
    train_loss, train_metrics = do_iteration(True)
    test_loss, test_metrics = do_iteration(False)
    print(f'epoch:{epoch} train {train_loss:.6e} {train_metrics["token_accuracy"]:.4f} {train_metrics["precision"]:.4f}/{train_metrics["recall"]:.4f}/{train_metrics["f-score"]:.4f}')
    print(f'epoch:{epoch} test  {test_loss:.6e} {test_metrics["token_accuracy"]:.4f} {test_metrics["precision"]:.4f}/{test_metrics["recall"]:.4f}/{test_metrics["f-score"]:.4f}')
    if best_test_metrics["f-score"] < test_metrics["f-score"]:
        best_test_metrics = test_metrics
        best_test_metrics["epoch"] = epoch
        best_test_metrics["loss"] = test_loss
        print("best ->", best_test_metrics)
print(best_test_metrics)

