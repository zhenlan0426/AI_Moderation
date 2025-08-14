# %% [code]
# %% [code] {"execution":{"iopub.status.busy":"2025-08-14T10:46:15.693590Z","iopub.execute_input":"2025-08-14T10:46:15.694212Z","iopub.status.idle":"2025-08-14T10:46:38.592959Z","shell.execute_reply.started":"2025-08-14T10:46:15.694186Z","shell.execute_reply":"2025-08-14T10:46:38.592233Z"}}
import os
from typing import Dict, List
from unsloth import FastLanguageModel
from peft import PeftModel
import torch.multiprocessing as mp
import torch
import pandas as pd
from collections import defaultdict
import sys
from utility import normalize_text_columns, build_dataloader_map
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('/kaggle/usr/lib/utility')

# %% [code]
model_name = "/kaggle/input/basemodel/transformers/default/1/base_model"
lora_dir = "/kaggle/input/lora-jigsaw/merged_model4b/merged_model4b"
lm_head_path = "/kaggle/input/lora-jigsaw/lm_head_weight.pth"

# %% [code] {"execution":{"iopub.status.busy":"2025-08-14T10:49:16.642063Z","iopub.execute_input":"2025-08-14T10:49:16.642395Z","iopub.status.idle":"2025-08-14T10:49:16.826049Z","shell.execute_reply.started":"2025-08-14T10:49:16.642368Z","shell.execute_reply":"2025-08-14T10:49:16.825098Z"}}
# %%writefile inference.py
@torch.no_grad()
def _infer_on_split(
    split_df: pd.DataFrame,
    gpu_index: int,
    model_name: str,
    lora_dir: str,
    lm_head_path: str,
):
    """Worker process: run inference on one GPU and return per-row aggregates.
    Args:
        split_df: DataFrame with test data split for this GPU.
        gpu_index: Index of the GPU to bind this process to.
        model_name: Path to the base model.
        lora_dir: Path to the LoRA weights directory.
        lm_head_path: Path to the trained 2-class head weights.
    Returns:
        Dict[int, List[List[float]]]: Mapping from row_id to list of logits for each
    """
    # Bind this process to a specific GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    torch.cuda.set_device(0)  # becomes this process's GPU 0 after masking
    device = torch.device("cuda:0")

    # Load model/tokenizer on this GPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        # dtype = torch.float16, # TODO: try this
        load_in_4bit = True,
        device_map={"": 0},
    )
    model = PeftModel.from_pretrained(model, lora_dir, is_trainable=False, device_map={"": 0})
    FastLanguageModel.for_inference(model)
    model.eval()

    # Load trained 2-class head (No, Yes) with shape (hidden, 2)
    lm_head_weight = torch.load(lm_head_path, map_location="cpu").to(dtype=torch.float16).to(device)


    # Build dataloader; note: map dataset duplicates each row (2 variants)
    dataloader = build_dataloader_map(
        df=split_df,
        tokenizer=tokenizer,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        include_body=False,
    )

    row_id_to_list = defaultdict(list)
    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for row_id, input_ids, vi_index, _ in dataloader:
            # Tensors are shaped: input_ids (1, L). vi_index is [first_vi_end, second_vi_end]
            input_ids = input_ids.to(device)
            output = model.base_model.model.model(input_ids)
            logits = output.last_hidden_state[0, -1] @ lm_head_weight
            # Use the last Violation: (target comment). Take probability of Yes (index 1)
            row_id_to_list[row_id].append(logits.detach().to("cpu").tolist())

    return row_id_to_list

@torch.no_grad()
def sum_softmax(logits):
    logits = torch.tensor(logits, dtype=torch.float32)
    return float(torch.softmax(torch.mean(logits, dim=0), dim=0)[1].item())

def aggregate_predictions(row_id_to_list: Dict[int, List[float]],
                            fn) -> Dict[int, float]:
    """Aggregate predictions for each row_id using the specified function.
    Args:
        row_id_to_list: Mapping from row_id to list of logits of shape (predictions, 2).
        fn: Function to aggregate logits (e.g., mean, max).
    Returns:
        Dict[int, float]: Mapping from row_id to aggregated prediction.
    """
    return [[row_id, fn(logits)] for row_id, logits in row_id_to_list.items()]

# %% [code] {"execution":{"iopub.status.busy":"2025-08-14T14:03:47.571051Z","iopub.execute_input":"2025-08-14T14:03:47.571282Z","iopub.status.idle":"2025-08-14T14:03:47.580810Z","shell.execute_reply.started":"2025-08-14T14:03:47.571258Z","shell.execute_reply":"2025-08-14T14:03:47.579749Z"}}
# from inference import *
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    dtypes = {"row_id": int,
                "body": str,
                "rule": str,
                "subreddit": str,
                "positive_example_1": str,
                "positive_example_2": str,
                "negative_example_1": str,
                "negative_example_2": str,
                }
    test_df = pd.read_csv("/kaggle/input/jigsaw-agile-community-rules/test.csv",
                        usecols=list(dtypes.keys()),  
                        dtype=dtypes)
    test_df = normalize_text_columns(test_df)
    
    # Randomly split into two halves
    test_df = test_df.sample(frac=1.0).reset_index(drop=True)
    mid = len(test_df) // 2
    split_0 = test_df.iloc[:mid].reset_index(drop=True)
    split_1 = test_df.iloc[mid:].reset_index(drop=True)
    
    # Spawn 2 worker processes, each bound to one GPU
    args_list = [(df_part, gpu_idx, model_name, lora_dir, lm_head_path)
                 for gpu_idx, df_part in enumerate([split_0, split_1])]
    
    # Create a Pool and use starmap to execute the function with multiple arguments
    with mp.Pool(processes=len(args_list)) as pool:
        results = pool.starmap(_infer_on_split, args_list)
    
    
    results[0].update(results[1])  # Merge results from both halves
    results = aggregate_predictions(results[0], sum_softmax)
    sub = pd.DataFrame(results, columns=["row_id", "rule_violation"])
    sub.to_csv("submission.csv", index=False)

# %% [code]
