#!uv run deepspeed --include localhost:1,2 main.deepspeed.pipe.py --deepspeed --deepspeed_config ./deepspeed.pipe.json --pipeline_size 2 --reduce_memory
# %%
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import time

import numpy as np
import torch

import deepspeed
import deepspeed.pipe

# %%
deepspeed.init_distributed(dist_backend='nccl')
import argparse
parser = argparse.ArgumentParser()
parser = deepspeed.add_config_arguments(parser)
parser.add_argument("--local_rank", type=int) # deepspeed requirement (precisely, torch's distributed running)

# user defined args
parser.add_argument("--pipeline_size", type=int, default=None, help="if specified, use pipeline parallel. must be non-negative.")
parser.add_argument("--reduce_memory", action="store_true", help="prepare initial model by LayerSpec. Only applicable when using pipeline")

args = parser.parse_args()
DO_PIPELINE = (args.pipeline_size is not None)

# %%
INPUT_DIM = 16
OUTPUT_DIM = 128
DATASET_SIZE = 96

class Dataset(torch.utils.data.Dataset):
    def __init__(self, size, input_dim=INPUT_DIM):
        rng = np.random.RandomState(seed=42)
        x_data = rng.normal(0.0, 1.0, size=[size, input_dim])
        x_data[:,0] = np.arange(size, dtype=x_data.dtype)
        self.x = torch.tensor(x_data, dtype=torch.bfloat16)
        self.y = torch.tensor(rng.randint(OUTPUT_DIM, size=[size]), dtype=torch.long)

    def __len__(self):
        return self.x.shape[0]
    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
data_set = Dataset(size=DATASET_SIZE)

# %%
HIDDEN_DIMS = [512, 4096, 8192, 16384, 32768, 16384, 8192, 8192, 4096]
# HIDDEN_DIMS = [512, 4096, 8192, 16384, 32768, 65536, 32768, 16384, 8192, 8192, 4096] # not fit 2x A100 w/o pipeline parallelism

ce_criterion = torch.nn.CrossEntropyLoss()

class LinearWithPrint(torch.nn.Linear):
    def forward(self, inputs):
        print(f"From LinearWithPrint: batch_indices={list(map(int, inputs[:,0].tolist()))}")
        return super().forward(inputs)

class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()

        dim_seqs = [INPUT_DIM] + HIDDEN_DIMS + [OUTPUT_DIM]
        self.layers = list()

        if args.reduce_memory and DO_PIPELINE:
            self.callable = False

            self.layers.append(deepspeed.pipe.LayerSpec(LinearWithPrint, dim_seqs[0], dim_seqs[1]))
            for i in range(1, len(dim_seqs)-1):
                self.layers.append(deepspeed.pipe.LayerSpec(torch.nn.ReLU))
                self.layers.append(deepspeed.pipe.LayerSpec(torch.nn.Linear, dim_seqs[i], dim_seqs[i+1]))

        else:
            self.callable = True

            self.layers.append(LinearWithPrint(dim_seqs[0], dim_seqs[1]))
            for i in range(1, len(dim_seqs)-1):
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Linear(dim_seqs[i], dim_seqs[i+1]))
            self.layers = torch.nn.Sequential(*self.layers)

    def forward(self, x, y):
        if not self.callable: raise ValueError("initialized by LayerSpec")
        logits = self.layers(x)
        loss = ce_criterion(logits, y)
        return loss

print("load model")
model = Model()

def loss_fn(model_outputs, labels):
    loss = ce_criterion(model_outputs, labels)
    print("loss:", loss.item(), "shape", labels.shape)
    return loss

if DO_PIPELINE:
    model = deepspeed.pipe.PipelineModule(layers=model.layers, num_stages=args.pipeline_size, loss_fn=loss_fn)

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
    str(next(iter(data_loader))[0].shape),
    "",
    "sleep 3 secs",
    "",
]), flush=True)
time.sleep(3)

if not DO_PIPELINE:
    EPOCH = 1
    for _ in range(EPOCH):
        for minibatch in data_loader:
            x, y = minibatch
            x, y = map(lambda x: x.to(model_engine.device), [x, y])
            print("\n".join([
                "----- input shape -----",
                str(x.shape),
            ]))

            loss = model_engine(x, y)
            print("----- loss -----")
            print(loss)
            print()

            model_engine.backward(loss)
            model_engine.step()


else:
    NUM_UPDATES = 6
    train_iter = iter(data_loader)
    for _ in range(NUM_UPDATES):
        loss = model_engine.train_batch(data_iter=train_iter)
        print("----- loss -----")
        print(loss)



# %%

# %%
import time
print("sleep 3 secs")
time.sleep(3)


# %%
