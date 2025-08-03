# %%
from unsloth import FastModel
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_value_
import time
from collections import defaultdict
from utility import build_dataloader
import pandas as pd
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model, tokenizer = FastModel.from_pretrained(
    # model_name = "unsloth/Qwen3-8B-unsloth-bnb-4bit",
    model_name = "unsloth/Qwen3-8B-Base-unsloth-bnb-4bit",
    # model_name="unsloth/gemma-3-12b-pt",
    # model_name="unsloth/gemma-3-4b-pt",
    max_seq_length = 8192, # Choose any for long context!
    load_in_4bit = True,
)

# %%
# # save the lm_head for yes and no
# index = [tokenizer.encode(" No")[0], tokenizer.encode(" Yes")[0]]
# print(index)
# torch.save(model.lm_head.weight[index], './Model/Gwen8B_lm_head_base.pth')
lm_head_weight = torch.load('./Model/Gwen8B_lm_head_base.pth').T
# lm_head_weight = torch.load('./Model/Gwen8B_lm_head.pth').T
lm_head_weight.requires_grad_(True);

# %%
# load dataset
df = pd.read_csv("Data/Data1/train.csv")
dataloader = build_dataloader(df, tokenizer, num_workers=4)

# %% [markdown]
# #### Fine-tune lm_head

# %%
epochs = 3
accumulation_steps = 64
lr = 1e-6
clip = 1e-4

# %%
trainable_params = [lm_head_weight]
optimizer = torch.optim.AdamW(trainable_params,lr = lr)
loss_fn = torch.nn.CrossEntropyLoss()
print(len(trainable_params))

# %%
start_time = time.time()
train_loss = 0
prob_list = defaultdict(list)
for epoch in range(epochs):
    for i, (row_id, input_ids, vi_index, labels) in enumerate(dataloader):
        row_id = int(row_id)
        with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids, vi_index, labels = input_ids.to('cuda'), vi_index.to('cuda'), labels.to('cuda')
            with torch.no_grad(): # as we are training the lm_head only.
                output = model.model(input_ids)
            logits = output.last_hidden_state[0, vi_index] @ lm_head_weight # (# of Violation, 4096) @ (4096, 2) -> (# of Violation, 2)
            loss = loss_fn(logits[:2], labels) # first 2 tokens are used for training, (N, C), (N,)
            loss.backward()
            train_loss += loss.item()
            if vi_index.shape[0] == 3:
                prob = torch.softmax(logits[2], dim=0)[1].item()
                prob_list[row_id].append(prob)
            if (i + 1) % accumulation_steps == 0:
                clip_grad_value_(trainable_params,clip)
                optimizer.step()
                optimizer.zero_grad()
    print(f"Epoch {epoch} loss: {train_loss / (i+1)}")


# %%



