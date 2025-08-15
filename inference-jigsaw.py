import os
from typing import Dict, List
import multiprocessing as mp  # use stdlib to avoid importing torch in child processes
import pandas as pd
from collections import defaultdict
import sys
import pickle
from utility import normalize_text_columns, build_dataloader_map
os.environ["TOKENIZERS_PARALLELISM"] = "false"
sys.path.append('/kaggle/usr/lib/utility')

# Add configurable environment flag: set IS_LOCAL=1 to run local single-GPU inference.
IS_LOCAL = True
Is_DEBUG = True

# AMP_DTYPE will be defined lazily *after* torch is imported inside each worker / main process.

if IS_LOCAL:
    # ---------------------------------------------------------------------
    # Local paths (edit via environment variables if you store models elsewhere)
    # ---------------------------------------------------------------------
    model_name = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
    lora_dir   = "Model/merged_model4b"
    lm_head_path = 'Model/lm_head_weight.pth'
else:
    # ---------------------------------------------------------------------
    # Kaggle competition paths – unchanged
    # ---------------------------------------------------------------------
    model_name = "/kaggle/input/basemodel/transformers/default/1/base_model"
    lora_dir   = "/kaggle/input/lora-jigsaw/merged_model4b/merged_model4b"
    lm_head_path = "/kaggle/input/lora-jigsaw/lm_head_weight.pth"

# Heavy libraries are imported *inside* this function after GPU masking so that
# each spawned worker only sees its assigned GPU.
def _infer_on_split(
    split_df: pd.DataFrame,
    gpu_index: int,
    model_name: str,
    lora_dir: str,
    lm_head_path: str,
    grouped_examples: Dict[str, Dict[str, List[str]]] | None = None,
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
    # ------------------------------------------------------------------
    # GPU masking *must* happen before importing torch / CUDA libraries.
    # ------------------------------------------------------------------
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Local (inside-worker) imports: no heavy CUDA initialisation happened yet.
    import torch
    from unsloth import FastLanguageModel
    from peft import PeftModel

    # Select the (sole) visible GPU as index 0 and prepare dtype
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")
    AMP_DTYPE = torch.bfloat16 if IS_LOCAL else torch.float16

    # Load model/tokenizer on this GPU
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name,
        load_in_4bit = True,
        device_map={"": 0},            # 0 is correct after CUDA_VISIBLE_DEVICES masking
    )
    model = PeftModel.from_pretrained(
        model,
        lora_dir,
        is_trainable=False,
        device_map={"": 0},
    )
    FastLanguageModel.for_inference(model)
    model.eval()

    # Load trained 2-class head (No, Yes) – shape (hidden, 2)
    lm_head_weight = (
        torch.load(lm_head_path, map_location="cpu")
        .to(dtype=AMP_DTYPE)
        .to(device)
    )


    # Build dataloader; note: map dataset duplicates each row (2 variants)
    dataloader = build_dataloader_map(
        df=split_df,
        tokenizer=tokenizer,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        include_body=False,
        grouped_examples=grouped_examples,
    )

    row_id_to_list = defaultdict(list)
    with torch.amp.autocast(device_type="cuda", dtype=AMP_DTYPE):
        for row_id, input_ids, vi_index, _ in dataloader:
            # input_ids: (1, L); vi_index = [first_vi_end, second_vi_end]
            input_ids = input_ids.to(device)
            output = model.base_model.model.model(input_ids)
            # Use logits at the **second** "Violation:" token (target comment)
            logits = output.last_hidden_state[0, vi_index[1]] @ lm_head_weight
            row_id_to_list[int(row_id)].append(logits.detach().cpu().tolist())

    return row_id_to_list

def sum_softmax(logits):
    """Convert two-class logits list → scalar margin (Yes − No)."""
    import torch
    with torch.no_grad():
        logits = torch.tensor(logits, dtype=torch.float32)
        mean = torch.mean(logits, dim=0)
        return (mean[1] - mean[0]).item()

def aggregate_predictions(row_id_to_list: Dict[int, List[float]],
                            fn) -> Dict[int, float]:
    """Aggregate predictions for each row_id using the specified function.
    Args:
        row_id_to_list: Mapping from row_id to list of logits of shape (predictions, 2).
        fn: Function to aggregate logits (e.g., mean, max).
    Returns:
        Dict[int, float]: Mapping from row_id to aggregated prediction.
    """
    return [[int(row_id), fn(logits)] for row_id, logits in row_id_to_list.items()]

# ---------------------------------------------------------------------------
# Main entry point – executed only in the parent process (not in workers).  
# Heavy CUDA libraries are imported here because GPU masking is irrelevant for
# the *parent*; only child processes need strict isolation.
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import torch  # safe to import here – parent process can see all GPUs
    mp.set_start_method("spawn", force=True)
    AMP_DTYPE = torch.bfloat16 if IS_LOCAL else torch.float16
    dtypes = {"row_id": int,
                "body": str,
                "rule": str,
                "subreddit": str,
                "positive_example_1": str,
                "positive_example_2": str,
                "negative_example_1": str,
                "negative_example_2": str,
                }
    test_df = pd.read_csv(
        ("Data/Data1/test.csv" if IS_LOCAL else "/kaggle/input/jigsaw-agile-community-rules/test.csv"),
        usecols=list(dtypes.keys()),
        dtype=dtypes,
    )
    test_df = normalize_text_columns(test_df)
    
    # Decide how many GPUs/workers to use
    GPU_COUNT = 1 if IS_LOCAL else torch.cuda.device_count()

    if GPU_COUNT == 1:
        # -------------------------------------------------------------
        # Single-GPU/local execution – no multiprocessing or splitting
        # -------------------------------------------------------------
        results_dict = _infer_on_split(
            test_df,
            0,
            model_name,
            lora_dir,
            lm_head_path,
        )
        if Is_DEBUG:
            # save the results
            with open("results_dict.pkl", "wb") as f:
                pickle.dump(results_dict, f)
        results = aggregate_predictions(results_dict, sum_softmax)
    else:
        # -------------------------------------------------------------
        # Multi-GPU/Kaggle execution
        # -------------------------------------------------------------
        from transformers import AutoTokenizer
        from utility import group_examples_by_rule
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        grouped_examples=group_examples_by_rule(test_df, include_body=False, tokenizer=tokenizer)
        # Randomly shuffle & split roughly equally across available GPUs
        test_df = test_df.sample(frac=1.0).reset_index(drop=True)
        splits = torch.chunk(torch.tensor(test_df.index.values), GPU_COUNT)
        df_splits = [test_df.iloc[split.tolist()].reset_index(drop=True) for split in splits]

        args_list = [
            (df_part, gpu_idx, model_name, lora_dir, lm_head_path, grouped_examples)
            for gpu_idx, df_part in enumerate(df_splits)
        ]

        with mp.Pool(processes=len(args_list)) as pool:
            worker_results = pool.starmap(_infer_on_split, args_list)
        # merge results
        results_dict = worker_results[0]
        for d in worker_results[1:]:
            results_dict.update(d)
        if Is_DEBUG:
            # save the results
            with open("results_dict.pkl", "wb") as f:
                pickle.dump(results_dict, f)
        results = aggregate_predictions(results_dict, sum_softmax)
    
    # -------------------------------------------------------------
    # Save submission file
    # -------------------------------------------------------------
    sub = pd.DataFrame(results, columns=["row_id", "rule_violation"])
    sub.to_csv("submission.csv", index=False)
