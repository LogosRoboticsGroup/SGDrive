# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import numpy as np
import torch

IGNORE_INDEX = -100


def pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
    return batch


def concat_pad_data_collator(features, max_item_length=None, pad_id=0):

    first = features[0]
    batch = {}

    batch_lens = [feat["input_ids"].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat["input_ids"].shape[0]] = feat["input_ids"]
        feat["input_ids"] = temp_input_ids
        temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat["labels"].shape[0]] = feat["labels"]
        feat["labels"] = temp_labels
        feat["attention_mask"] = feat["input_ids"].ne(pad_id)

        if "position_ids" in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[: feat["position_ids"].shape[0]] = feat["position_ids"]
            feat["position_ids"] = temp_position_ids

        if "loss_weight" in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[: feat["loss_weight"].shape[0]] = feat["loss_weight"]
            feat["loss_weight"] = temp_loss_weight

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = (
            first["label"].item()
            if isinstance(first["label"], torch.Tensor)
            else first["label"]
        )
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = (
                torch.long if isinstance(first["label_ids"][0], int) else torch.float
            )
            batch["labels"] = torch.tensor(
                [f["label_ids"] for f in features], dtype=dtype
            )

    # for 3d bbox
    if "agent_states" in first:
        max_num_agents = max(f['agent_states'].shape[0] if f['agent_states'].numel() > 0 else 0 for f in features)
        agent_state_dim = 7

        padded_states = []
        padded_labels = []
        for f in features:
            if f['agent_states'].numel() == 0:
                state_pad = torch.zeros((max_num_agents, agent_state_dim))
                label_pad = torch.full((max_num_agents,), -1, dtype=torch.long)
                padded_states.append(state_pad)
                padded_labels.append(label_pad)
            else:
                n = f['agent_states'].shape[0]
                pad_n = max_num_agents - n
                state_pad = torch.zeros((pad_n, agent_state_dim))
                label_pad = torch.full((pad_n,), -1, dtype=torch.long)
                padded_states.append(torch.cat([f['agent_states'], state_pad], dim=0))
                padded_labels.append(torch.cat([f['agent_labels'], label_pad], dim=0))

        batch["agent_states"] = torch.stack(padded_states)  # (B, max_agents, state_dim)
        batch["agent_labels"] = torch.stack(padded_labels)  # (B, max_agents)
    
       #for 3d bbox
    if 'agent_states_ft' in first:
        max_num_agents = max(f['agent_states_ft'].shape[0] if f['agent_states_ft'].numel() > 0 else 0 for f in features)
        agent_state_dim = 7

        padded_states_ft = []
        padded_labels_ft = []
        for f in features:
            if f['agent_states_ft'].numel() == 0:
                state_pad = torch.zeros((max_num_agents, agent_state_dim))
                label_pad = torch.full((max_num_agents,), -1, dtype=torch.long)
                padded_states_ft.append(state_pad)
                padded_labels_ft.append(label_pad)
            else:
                n = f['agent_states_ft'].shape[0]
                pad_n = max_num_agents - n
                state_pad = torch.zeros((pad_n, agent_state_dim))
                label_pad = torch.full((pad_n,), -1, dtype=torch.long)
                padded_states_ft.append(torch.cat([f['agent_states_ft'], state_pad], dim=0))
                padded_labels_ft.append(torch.cat([f['agent_labels_ft'], label_pad], dim=0))

        batch['agent_states_ft'] = torch.stack(padded_states_ft)  # (B, max_agents, state_dim)
        batch['agent_labels_ft'] = torch.stack(padded_labels_ft)  # (B, max_agents)

    if 'lidar_gt' in first:
        dtype = torch.long 
        batch['lidar_gt'] = torch.tensor([f['lidar_gt'] for f in features], dtype=dtype)
    if 'lidar_gt_dream' in first:
        dtype = torch.long 
        batch['lidar_gt_dream'] = torch.tensor([f['lidar_gt_dream'] for f in features], dtype=dtype)
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if (
            k
            not in (
                "label",
                "label_ids",
                "pixel_values",
                "image_flags",
                "agent_labels",
                "agent_states",
                "lidar_gt",
                "lidar_gt_dream",
                "agent_labels_ft",
                "agent_states_ft",
            )
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("pixel_values", "image_flags"):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


def dpo_concat_pad_data_collator(features, pad_id=0):

    first = features[0]
    batch = {}

    for prefix in ["chosen_", "rejected_"]:
        batch_lens = [feat[f"{prefix}input_ids"].shape[0] for feat in features]
        max_item_length = max(batch_lens)
        for idx in range(len(features)):
            feat = features[idx]
            temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_input_ids[: feat[f"{prefix}input_ids"].shape[0]] = feat[
                f"{prefix}input_ids"
            ]
            feat[f"{prefix}input_ids"] = temp_input_ids
            temp_labels = torch.LongTensor([IGNORE_INDEX] * max_item_length)
            temp_labels[: feat[f"{prefix}labels"].shape[0]] = feat[f"{prefix}labels"]
            feat[f"{prefix}labels"] = temp_labels
            feat[f"{prefix}attention_mask"] = feat[f"{prefix}input_ids"].ne(pad_id)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if (
            k not in ("pixel_values", "image_flags")
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ("pixel_values", "image_flags"):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch
