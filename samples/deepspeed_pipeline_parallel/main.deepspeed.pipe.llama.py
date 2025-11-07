#!uv run deepspeed --include localhost:4,5,6,7 main.deepspeed.pipe.llama.py --deepspeed --deepspeed_config ./deepspeed.pipe.json --pipeline_size 4 --sequence_length 4096 --inherit move
# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time
import json
import functools

import numpy as np
import torch
import transformers

import deepspeed
import deepspeed.pipe

import modeling_llama

# %%
deepspeed.init_distributed(dist_backend='nccl')
import argparse
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int) # deepspeed requirement (precisely, torch's distributed running)

# user defined args
parser.add_argument("--pipeline_size", type=int, default=None, help="if specified, use pipeline parallel. must be non-negative.")

parser.add_argument("--sequence_length", "-s", type=int, default=32)
parser.add_argument("--inherit", type=str, default="move", choices=["move", "copy", "init"])
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")

args = parser.parse_args()
DO_PIPELINE = (args.pipeline_size is not None)

# %%
DATASET_SIZE = 512

class Dataset(torch.utils.data.Dataset):
    def __init__(self, size, seq_len):
        rng = np.random.RandomState(seed=42)
        # tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name)
        # input_ids = tokenizer.apply_chat_template([{"role":"user","content":"hello!"},{"role":"assistant","content":"ABCDE"}], return_tensors="pt")
        with open("token_ids.json") as f:
            input_ids = torch.tensor(json.load(f)["llama-3.1"], dtype=torch.long)
        if input_ids.shape[1] < seq_len: raise ValueError("not enough length")
        self.input_ids = input_ids[:1,:seq_len].tile(size,1)
        self.attention_mask = torch.ones_like(self.input_ids, dtype=torch.float32)
        self.labels = input_ids[:1,1:seq_len+1].tile(size,1)
    def __len__(self):
        return self.input_ids.shape[0]
    def __getitem__(self, idx):
        return (self.input_ids[idx], self.attention_mask[idx]), self.labels[idx]
data_set = Dataset(size=DATASET_SIZE, seq_len=args.sequence_length)

# %%
print("load model")
if args.inherit in ["move", "copy"]:
    # base_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model = transformers.AutoModelForCausalLM.from_pretrained(args.model_name, device_map="cpu", torch_dtype="bfloat16", attn_implementation="flash_attention_2")
    converted_model = modeling_llama.convert(base_model.config, base_model, inherit=args.inherit)
elif args.inherit == "init":
    model_config = transformers.AutoConfig.from_pretrained(args.model_name)
    model_config._attn_implementation = "flash_attention_2"
    converted_model = modeling_llama.convert(model_config, inherit=args.inherit)

ce_criterion = torch.nn.CrossEntropyLoss()
def loss_fn(logits, labels):
    loss = ce_criterion(logits.reshape(-1, logits.shape[-1]), labels.reshape(-1))
    print("loss:", loss.item(), "labels shape", labels.shape)
    return loss

if DO_PIPELINE:
    model = deepspeed.pipe.PipelineModule(
        layers=converted_model, num_stages=args.pipeline_size, loss_fn=loss_fn,
        # activation_checkpoint_interval=0, activation_checkpoint_func=functools.partial(torch.utils.checkpoint.checkpoint,use_reentrant=False),
    )
else:
    class Mod(torch.nn.Module):
        def __init__(self, base):
            super().__init__()
            self.base = base
        def forward(self, x, y):
            logits = self.base(x)
            loss = loss_fn(logits, y)
            return loss
    model = Mod(converted_model)


# %%
# deepspeed.init_distributed()
model_engine, optimizer, data_loader, _ = deepspeed.initialize(
    args=args,
    model=model,
    model_parameters=model.parameters(),
    training_data=data_set,
)


# %%
print("\n".join([
    "----- input shape (by iter(data_loader)) -----",
    str(next(iter(data_loader))[0][0].shape),
    "",
    "sleep 3 secs",
    "",
]), flush=True)
time.sleep(3)
start_time = time.perf_counter()

NUM_UPDATES = 6
if not DO_PIPELINE:
    update_count = 0
    for _ in range(NUM_UPDATES):
        for minibatch in data_loader:
            if model_engine.is_gradient_accumulation_boundary():
                update_count += 1

            x, y = minibatch
            *x, y = map(lambda a: a.to(model_engine.device), [*x, y])
            print("\n".join([
                "----- input shape -----",
                str(x[0].shape),
            ]))

            loss = model_engine(x, y)
            print("----- loss -----")
            print(loss)
            print()

            model_engine.backward(loss)
            model_engine.step()

            if update_count >= NUM_UPDATES:
                break
        if update_count >= NUM_UPDATES:
            break


else:
    train_iter = iter(data_loader)
    for _ in range(NUM_UPDATES):
        loss = model_engine.train_batch(data_iter=train_iter)
        print("----- loss -----")
        print(loss)



# %%

# %%
end_time = time.perf_counter()
print(f'takes {end_time-start_time:.1f} [sec]')

print("sleep 3 secs")
time.sleep(3)


# %%
