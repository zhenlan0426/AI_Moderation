from __future__ import annotations

import argparse
import os
import pickle
from typing import Dict, List, Tuple, Iterable, Any
import math

import torch
from unsloth import FastLanguageModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Local import
from utility import load_grouped_data


def _ensure_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, exist_ok=True)


def _sanitize_text_list(texts: List[Any]) -> List[str]:
    """Return a cleaned list of strings, dropping None/NaN/empty items.

    Ensures the tokenizer always receives well-formed strings.
    """
    cleaned: List[str] = []
    for item in texts:
        if item is None:
            continue
        # Drop float NaN
        if isinstance(item, float) and math.isnan(item):
            continue
        s = str(item).strip()
        if not s:
            continue
        if s.lower() in {"nan", "none"}:
            continue
        cleaned.append(s)
    return cleaned


def encode_texts_batch(texts: List[Any], tokenizer) -> List[List[int]]:
    """Encode a list of strings into lists of token ids without special tokens.

    Returns list of list of ints (ragged). This keeps compatibility with
    downstream functions that expect Python lists.
    """
    cleaned_texts = _sanitize_text_list(texts)
    if len(cleaned_texts) == 0:
        return []
    enc = tokenizer.batch_encode_plus(
        cleaned_texts,
        add_special_tokens=False,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    return enc["input_ids"]


def tokenize_grouped_data(
    grouped: Dict[str, Dict[str, List[str]]],
    tokenizer,
) -> Tuple[Dict[str, Dict[str, List[List[int]]]], List[List[int]]]:
    """Tokenize the grouped dataset.

    Maintains sharing of the negatives list if the input uses the same object
    across rules by caching encodings keyed by Python object id.

    Returns the tokenized grouped structure and an aggregate list of all sequences
    (list of list[int]) for downstream unique-id computation.
    """
    tokenized: Dict[str, Dict[str, List[List[int]]]] = {}
    cache_by_obj_id: Dict[int, List[List[int]]] = {}

    # Only add each pool's sequences once (important for shared negatives)
    seen_pool_obj_ids: set[int] = set()
    all_sequences: List[List[int]] = []

    for rule, pools in grouped.items():
        tokenized_rule: Dict[str, List[List[int]]] = {}
        for polarity in ("positives", "negatives"):
            leaf_list = pools.get(polarity, [])
            obj_key = id(leaf_list)
            if obj_key not in cache_by_obj_id:
                cache_by_obj_id[obj_key] = encode_texts_batch(leaf_list, tokenizer)
            encoded = cache_by_obj_id[obj_key]
            tokenized_rule[polarity] = encoded
            if obj_key not in seen_pool_obj_ids:
                all_sequences.extend(encoded)
                seen_pool_obj_ids.add(obj_key)
        tokenized[rule] = tokenized_rule

    return tokenized, all_sequences


def compute_unique_token_ids(
    sequences: List[List[int]],
) -> torch.Tensor:
    """Compute unique token ids across all sequences using torch.

    Returns a 1D torch.LongTensor of unique ids.
    """
    chunks = [torch.unique(torch.as_tensor(seq, dtype=torch.long)) for seq in sequences]
    all_ids = torch.cat(chunks, dim=0)
    uniq = torch.unique(all_ids)
    return uniq


def build_vocab_mapping(
    used_old_ids: torch.Tensor,
    original_vocab_size: int,
) -> Tuple[torch.Tensor, int]:
    """Build mapping tensors between old and new vocabularies.

    Expects `used_old_ids` to be a 1D LongTensor of unique ids.
    Special-token handling is omitted; we only provide a UNK fallback if present.
    Returns (old_to_new, new_to_old, kept_old_ids_in_new_order, new_unk_index).
    """
    # New vocab order is exactly the set of used ids (already unique and long)
    print("assume tokenizer does not have special tokens - UNK")
    new_to_old = used_old_ids
    
    # Fallback UNK index is appended position: new_vocab_size
    new_vocab_size = int(new_to_old.numel())
    
    # Build mapping tensor pre-filled with fallback (points to appended UNK row)
    old_to_new = torch.full((original_vocab_size,), new_vocab_size, dtype=torch.long)
    old_to_new.index_copy_(0, new_to_old, torch.arange(new_vocab_size, dtype=torch.long))

    return old_to_new, new_vocab_size


def remap_grouped_token_ids(
    tokenized_grouped: Dict[str, Dict[str, List[List[int]]]],
    old_to_new: torch.Tensor,
) -> Dict[str, Dict[str, List[torch.Tensor]]]:
    """Remap nested grouped token ids using an ``old_to_new`` id map.

    Accepts leaves as ``List[List[int]]`` for compatibility with existing code,
    converts leaves to 1D Long tensors internally for vectorised remapping, and
    returns the same structure with leaves as tensors. Shared pool lists are
    preserved by caching on the pool list object's id.
    """

    # Cache remapped leaves by original list object id to preserve sharing.
    remapped_by_obj_id: Dict[int, List[torch.Tensor]] = {}

    def remap_sequence(sequence_ids: List[int]) -> torch.Tensor:
        seq = torch.as_tensor(sequence_ids, dtype=torch.long, device=old_to_new.device)
        return old_to_new.index_select(0, seq)

    remapped: Dict[str, Dict[str, List[torch.Tensor]]] = {}
    for rule, pools in tokenized_grouped.items():
        remapped_rule: Dict[str, List[torch.Tensor]] = {}
        for polarity in ("positives", "negatives"):
            sequences = pools.get(polarity, [])
            obj_key = id(sequences)
            if obj_key not in remapped_by_obj_id:
                remapped_by_obj_id[obj_key] = [remap_sequence(seq) for seq in sequences]
            remapped_rule[polarity] = remapped_by_obj_id[obj_key]
        remapped[rule] = remapped_rule

    return remapped

def gather_reduced_embedding(
    model,
    new_to_old_ids: torch.Tensor,
    *,
    torch_dtype: torch.dtype | None = torch.float16,
) -> torch.Tensor:
    """Load model and gather the reduced input embedding by old indices.

    The returned tensor is on CPU.
    """

    emb = model.get_input_embeddings().weight.detach().to("cpu")
    reduced = emb.index_select(0, new_to_old_ids)
    return reduced


def save_pickle(obj: Any, path: str) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    # NOTE: there is a unidentifed remapping bug, resulting in worsening validation performance. and little improvement in VRAM. Disabled for now.
    from torch.utils.data import DataLoader
    from utility import TTTDataset_iter
    # model_name_or_path = "unsloth/Qwen3-4B-Base-unsloth-bnb-4bit"
    model_name_or_path = "unsloth/Qwen3-1.7B-Base-unsloth-bnb-4bit"
    data_dir = "Data/grouped"
    model_dir = "Model"
    _ensure_dir(data_dir)
    _ensure_dir(model_dir)
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_name_or_path,
        load_in_4bit = True,
    )

    # Load grouped datasets
    # NOTE: this no longer works, load_grouped_data and TTTDataset_iter was based on the old text tokenizer (Commit de7656c) generating the
    # remapped token ids. Now the code is changed to use token_ids directly.
    train_grouped, holdout_grouped = load_grouped_data(load_in_token=False)
    # # get unique token ids from the dataloader as there are some special tokens in the tokenizer or prompt
    # dataloader = DataLoader(
    #         TTTDataset_iter(train_grouped, holdout_grouped, tokenizer, samples_per_epoch=2000),
    #         batch_size=1,
    #         worker_init_fn=seed_worker,
    #         collate_fn=lambda x: x[0]
    #     )
    # sample_input_ids = []
    # for _, input_ids, _, _ in dataloader:
    #     sample_input_ids.append(input_ids)
    # Tokenize both splits
    train_tok, train_seqs = tokenize_grouped_data(train_grouped, tokenizer)
    hold_tok, hold_seqs = tokenize_grouped_data(holdout_grouped, tokenizer)

    # Save tokenized datasets
    save_pickle(train_tok, os.path.join(data_dir, "train_grouped_token_ids.pkl"))
    save_pickle(hold_tok, os.path.join(data_dir, "holdout_grouped_token_ids.pkl"))

    # # Compute unique token ids across both splits
    # all_unique_ids = compute_unique_token_ids(train_seqs + hold_seqs + sample_input_ids)
    # print(all_unique_ids.shape)

    # # Determine original vocab size from tokenizer
    # orig_vocab_size = tokenizer.vocab_size
    # # Build mapping tensors including special/UNK tokens
    # old_to_new, new_unk_index = build_vocab_mapping(
    #     used_old_ids=all_unique_ids,
    #     original_vocab_size=orig_vocab_size,
    #     )
    # remapped_train = remap_grouped_token_ids(train_tok, old_to_new)
    # remapped_hold = remap_grouped_token_ids(hold_tok, old_to_new)
    # # Save remapped datasets
    # save_pickle(remapped_train, os.path.join(data_dir, "train_grouped_token_ids_remapped.pkl"))
    # save_pickle(remapped_hold, os.path.join(data_dir, "holdout_grouped_token_ids_remapped.pkl"))

    # # Save mappings
    # torch.save(old_to_new, os.path.join(model_dir, "vocab_mapping.pt"))


    # emb_weight = model.get_input_embeddings().weight
    # orig_vocab_size = int(emb_weight.shape[0])


    # reduced_emb = gather_reduced_embedding(model, all_unique_ids, torch_dtype=torch.float16)
    # # Append fallback embedding row as the mean of kept embeddings (or zeros if none)
    # fallback_row = reduced_emb.mean(dim=0, keepdim=True)
    # reduced_emb = torch.cat([reduced_emb, fallback_row], dim=0)
    # torch.save(reduced_emb, os.path.join(model_dir, "reduced_embedding.pt"))


if __name__ == "__main__":
    main()


