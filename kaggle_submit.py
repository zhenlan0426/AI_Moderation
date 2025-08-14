#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import math
import time
from typing import Dict, Tuple, List
from unsloth import FastLanguageModel
from peft import PeftModel
import torch
import pandas as pd

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Project imports
from utility import normalize_text_columns, build_dataloader_map


@torch.no_grad()
def _infer_on_split(
    split_df: pd.DataFrame,
    gpu_index: int,
    results_dict,
    model_name: str = "/kaggle/input/basemodel/transformers/default/1/base_model",
    lora_dir: str = "/kaggle/input/lora-jigsaw/merged_model4b/merged_model4b",
    lm_head_path: str = "/kaggle/input/lora-jigsaw/lm_head_weight.pth",
):
    """Worker process: run inference on one GPU and return per-row aggregates.

    results_dict is a multiprocessing.Manager().dict() where we will place a tuple of
    (row_ids: List[int], sums: List[float], counts: List[int]) under key gpu_index.
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

    row_id_to_sum: Dict[int, float] = {}
    row_id_to_count: Dict[int, int] = {}


    with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
        for row_id, input_ids, vi_index, _ in dataloader:
            # Tensors are shaped: input_ids (1, L). vi_index is [first_vi_end, second_vi_end]
            input_ids = input_ids.to(device, non_blocking=True)
            output = model.base_model.model.model(input_ids)
            logits = output.last_hidden_state[0, -1] @ lm_head_weight
            # Use the last Violation: (target comment). Take probability of Yes (index 1)
            probs = torch.softmax(logits, dim=-1)
            prob_yes = float(probs[1].detach().to("cpu"))

            row_id_int = int(row_id)
            row_id_to_sum[row_id_int] = row_id_to_sum.get(row_id_int, 0.0) + prob_yes
            row_id_to_count[row_id_int] = row_id_to_count.get(row_id_int, 0) + 1

    # Convert to lists to make serialization cheaper
    ids: List[int] = list(row_id_to_sum.keys())
    sums: List[float] = [row_id_to_sum[i] for i in ids]
    counts: List[int] = [row_id_to_count[i] for i in ids]
    results_dict[gpu_index] = (ids, sums, counts)


def main() -> None:
    parser = argparse.ArgumentParser(description="Kaggle submission generator for AI Moderation")
    parser.add_argument("--test_csv", type=str, default="Data/Data1/test.csv")
    parser.add_argument("--output_csv", type=str, default="sample_submission.csv")
    parser.add_argument("--model_name", type=str, default="unsloth/Qwen3-4B-Base-unsloth-bnb-4bit")
    parser.add_argument("--lora_dir", type=str, default="Model/merged_model4b")
    parser.add_argument("--lm_head_path", type=str, default="Model/lm_head_weight.pth")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Load test CSV and normalize text columns
    test_df = pd.read_csv(args.test_csv)
    test_df = normalize_text_columns(test_df)

    # Randomly split into two halves
    test_df = test_df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)
    mid = len(test_df) // 2
    split_0 = test_df.iloc[:mid].reset_index(drop=True)
    split_1 = test_df.iloc[mid:].reset_index(drop=True)

    # Spawn 2 worker processes, each bound to one GPU
    from multiprocessing import get_context
    mp_ctx = get_context("spawn")
    manager = mp_ctx.Manager()
    shared_results = manager.dict()

    procs = []
    for gpu_idx, df_part in enumerate([split_0, split_1]):
        p = mp_ctx.Process(
            target=_infer_on_split,
            args=(df_part, gpu_idx, args.model_name, args.lora_dir, args.lm_head_path, shared_results),
        )
        p.start()
        procs.append(p)

    for p in procs:
        p.join()

    # Aggregate predictions across splits and per-row variants
    global_sum: Dict[int, float] = {}
    global_count: Dict[int, int] = {}
    for key in list(shared_results.keys()):
        ids, sums, counts = shared_results[key]
        for rid, s, c in zip(ids, sums, counts):
            global_sum[rid] = global_sum.get(rid, 0.0) + float(s)
            global_count[rid] = global_count.get(rid, 0) + int(c)

    # Build submission aligned to input test order (by row_id column)
    preds: List[float] = []
    for rid in test_df["row_id"].tolist():
        s = global_sum.get(int(rid), 0.0)
        c = global_count.get(int(rid), 0)
        prob = s / c if c > 0 else 0.5  # safe default
        preds.append(prob)

    sub = pd.DataFrame({"row_id": test_df["row_id"], "rule_violation": preds})
    sub.to_csv(args.output_csv, index=False)
    print(f"Wrote submission to {args.output_csv} with {len(sub)} rows.")


if __name__ == "__main__":
    main()


