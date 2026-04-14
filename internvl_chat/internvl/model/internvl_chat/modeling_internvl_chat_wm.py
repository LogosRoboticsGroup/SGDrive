# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------

import warnings
import os
from typing import List, Optional, Tuple, Union, Dict
import numpy as np
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM
from peft import LoraConfig, get_peft_model
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from transformers import (
    AutoModel,
    GenerationConfig,
    LlamaForCausalLM,
    LlamaTokenizer,
    Qwen2ForCausalLM,
)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging
from timm.models.layers import Mlp

from .configuration_internvl_chat_wm import InternVLChatConfig_WM
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .point_decoder import Decoder2D, WmEncoder
from .qwen_wrapper import Qwen_wrapper
import copy

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op="eq"):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


def linear_relu_ln(embed_dims, in_loops, out_loops, input_dims=None):
    if input_dims is None:
        input_dims = embed_dims
    layers = []
    for _ in range(out_loops):
        for _ in range(in_loops):
            layers.append(nn.Linear(input_dims, embed_dims))
            layers.append(nn.ReLU(inplace=True))
            input_dims = embed_dims
        layers.append(nn.LayerNorm(embed_dims))
    return layers

class InternVLChatModel_WM(PreTrainedModel):
    config_class = InternVLChatConfig_WM
    main_input_name = "pixel_values"
    base_model_prefix = "language_model"
    _no_split_modules = [
        "InternVisionModel",
        "LlamaDecoderLayer",
        "InternLM2DecoderLayer",
        "Phi3DecoderLayer",
        "Qwen2DecoderLayer",
    ]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: InternVLChatConfig_WM,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = (
            "flash_attention_2" if use_flash_attn else "eager"
        )

        logger.info(f"num_image_token: {self.num_image_token}")
        logger.info(f"ps_version: {self.ps_version}")
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == "LlamaForCausalLM":
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "InternLM2ForCausalLM":
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Phi3ForCausalLM":
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == "Qwen2ForCausalLM":
                # self.language_model = Qwen2ForCausalLM(config.llm_config)
                self.language_model = Qwen_wrapper(config.llm_config)
            else:
                raise NotImplementedError(
                    f"{config.llm_config.architectures[0]} is not implemented."
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None

        # define world token and world query
        self.NUM_WORLD_TOKEN = config.agent_token_number + config.gp_token_number + config.occ_token_number
        self.agent_token_number = config.agent_token_number
        self.gp_token_number = config.gp_token_number
        self.occ_token_number = config.occ_token_number
        self.dream_world = os.getenv("dream_world", True) #config.dream_world
        self.world_token_id = None
        self.world_queries = nn.Parameter(
            torch.zeros(1, self.NUM_WORLD_TOKEN, llm_hidden_size)
        )  # One world token corresponds to one world query, as currently defined
        nn.init.normal_(self.world_queries, mean=0.0, std=0.02)
        if self.dream_world:
            self.dream_queries = nn.Parameter(
                torch.zeros(1, self.occ_token_number+self.agent_token_number, llm_hidden_size)
            )
            nn.init.normal_(self.dream_queries, mean=0.0, std=0.02)
        self.dream_token_id = None

        # self._tf_decoder = nn.TransformerDecoder(tf_decoder_layer, 6)
        self._agent_head = AgentHead(
            num_agents=self.agent_token_number,
            d_ffn=llm_hidden_size,
            d_model=llm_hidden_size,
        )
        self._gp_head = GoalPointHead(
            language_hidden_dim=llm_hidden_size,
            output_dim=2,
        )
        self._wm_encoder = WmEncoder(llm_hidden_size, 8, 4)
        base_channel = 128
        self._occ_head = Decoder2D(
            llm_hidden_size = llm_hidden_size,
            ch = base_channel, 
            out_ch = 32, 
            ch_mult = (1,2,4), 
            num_res_blocks = 2,
            attn_resolutions = (50,), 
            dropout = 0.0, 
            resamp_with_conv = True, 
            in_channels = 64,
            resolution = 100, 
            z_channels = base_channel * 2, 
            give_pre_end = False
        )

        if self.dream_world:
            self._wm_encoder_dream = WmEncoder(llm_hidden_size, 8, 4)
            self._occ_head_dream = Decoder2D(
                llm_hidden_size = llm_hidden_size,
                ch = base_channel, 
                out_ch = 32, 
                ch_mult = (1,2,4), 
                num_res_blocks = 2,
                attn_resolutions = (50,), 
                dropout = 0.0, 
                resamp_with_conv = True, 
                in_channels = 64,
                resolution = 100, 
                z_channels = base_channel * 2, 
                give_pre_end = False
            )

            self._agent_head_dream = AgentHead(
                num_agents=self.agent_token_number,
                d_ffn=llm_hidden_size,
                d_model=llm_hidden_size,
            )

        # self._agent_head.apply(safe_init_weights)
        # self._gp_head.apply(safe_init_weights)

        self.conv_template = get_conv_template(self.template)
        if hasattr(config, "system_message"):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora
            )

        self.ego_status_encoder = Mlp(
            in_features=11,
            hidden_features=config.hidden_size,
            out_features=llm_hidden_size,
            norm_layer=nn.LayerNorm,
        )
        self.reduce_fused_feature = nn.Linear(in_features=llm_hidden_size*2, out_features=llm_hidden_size)
        self.hist_traj_encoder = TrajEncoder(d_model=llm_hidden_size)

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=["attn.qkv", "attn.proj", "mlp.fc1", "mlp.fc2"],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == "InternLM2ForCausalLM":
            target_modules = [
                "attention.wqkv",
                "attention.wo",
                "feed_forward.w1",
                "feed_forward.w2",
                "feed_forward.w3",
            ]
        elif self.llm_arch_name == "Phi3ForCausalLM":
            target_modules = [
                "mlp.down_proj",
                "mlp.gate_up_proj",
                "self_attn.o_proj",
                "self_attn.qkv_proj",
            ]
        elif self.llm_arch_name in ["Qwen2ForCausalLM", "LlamaForCausalLM"]:
            target_modules = [
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "mlp.gate_proj",
                "mlp.down_proj",
                "mlp.up_proj",
            ]
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type="CAUSAL_LM",
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def interact_ego_status(self, query, ego_status):
        ego_status = ego_status.to(query.device).to(query.dtype)
        ego_status_feature = self.ego_status_encoder(ego_status)
        expand_ego_status_feature = ego_status_feature.unsqueeze(1).expand(-1, query.size(1), -1)
        query_feature_plus = torch.cat([query, expand_ego_status_feature], dim=-1)
        query_feature_plus = self.reduce_fused_feature(query_feature_plus)
        return query_feature_plus
    
    def interact_hist_traj(self, query, hist_traj):
        hist_traj = hist_traj.to(query.device).to(query.dtype)
        hist_traj_feat = self.hist_traj_encoder(hist_traj)
        hit_traj_feat_broadcast = hist_traj_feat.unsqueeze(1).expand(-1, query.size(1), -1)  # (B, N, hidden_dim)
        fused_feat = query + hit_traj_feat_broadcast
        return fused_feat

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        agent_states: Optional[torch.Tensor] = None,  # for 3dbbox
        agent_labels: Optional[torch.Tensor] = None,  # for 3dbbox
        agent_states_ft: Optional[torch.Tensor] = None,  # for 3dbbox
        agent_labels_ft: Optional[torch.Tensor] = None,  # for 3dbbox
        trajectory: Optional[torch.Tensor] = None,  # for trajectory
        goalpoint: Optional[torch.Tensor] = None,   # for GoalPoint
        ego_status: Optional[torch.Tensor] = None, 
        hist_traj: Optional[torch.Tensor] = None, 
        lidar_gt: Optional[torch.Tensor] = None,   # for Occ
        lidar_gt_dream: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, dynamic ViT size per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = (
                    statistics.tolist()
                )
                self.num_samples += num_samples
                print(
                    f"total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}"
                )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        # replace world token with world query
        if self.world_token_id is not None:
            selected = input_ids == self.world_token_id
            world_embeds = self.world_queries.repeat(B, 1, 1)  #
            world_embeds = self._wm_encoder(vit_embeds, world_embeds)
            world_embeds = self.interact_ego_status(world_embeds, ego_status)
            world_embeds = self.interact_hist_traj(world_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + world_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                world_embeds = world_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"world_embeds.shape={world_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + world_embeds[:n_token]
                )
                ignore_flag = True
        
        if self.dream_token_id is not None:
            selected = input_ids == self.dream_token_id
            dream_embeds = self.dream_queries.repeat(B, 1, 1)  #
            dream_embeds = self._wm_encoder_dream(vit_embeds, dream_embeds)
            dream_embeds = self.interact_ego_status(dream_embeds, ego_status)
            dream_embeds = self.interact_hist_traj(dream_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + dream_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                dream_embeds = dream_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"dream_embeds.shape={dream_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + dream_embeds[:n_token]
                )
                ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        attention_mask = self.generate_attention_mask(input_embeds, input_ids)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )
        logits = outputs.logits
        if self.world_token_id is not None:
            selected = input_ids == self.world_token_id
            last_hidden_state = outputs.hidden_states[-1].view(-1, C)
            world_hidden = last_hidden_state[selected].view(B, -1, C).contiguous()
            
            if self.occ_token_number > 0:
                occ_world_hidden = world_hidden[:, : self.occ_token_number, :]
                cls_pred = self._occ_head(occ_world_hidden, [[100, 100], [50, 50]])
                cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2)
                occ_loss = self._occ_head.loss(cls_pred, lidar_gt)
            
            if self.agent_token_number > 0:
                agent_world_hidden = world_hidden[:, self.occ_token_number: self.occ_token_number+self.agent_token_number, :]
                agent_out = self._agent_head(agent_world_hidden)
                tgt = {"agent_states": agent_states, "agent_labels": agent_labels}
                det_loss = self._agent_loss_v1(tgt, agent_out)
                agent_loss = det_loss[0] + det_loss[1]
            
            if self.gp_token_number > 0:
                gp_world_hidden = world_hidden[:,self.occ_token_number+self.agent_token_number: self.occ_token_number+self.agent_token_number+self.gp_token_number, :]
                gp_out = self._gp_head(gp_world_hidden)
                gp_loss = self._gp_head.loss(gp_out, goalpoint)
        
        if self.dream_token_id is not None:
            selected = input_ids == self.dream_token_id
            last_hidden_state = outputs.hidden_states[-1].view(-1, C)
            dream_hidden = last_hidden_state[selected].view(B, -1, C).contiguous()
            
            if self.occ_token_number > 0:
                occ_dream_hidden = dream_hidden[:, : self.occ_token_number, :]
                cls_pred = self._occ_head_dream(occ_dream_hidden, [[100, 100], [50, 50]])
                cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2)
                occ_loss_dream = self._occ_head.loss(cls_pred, lidar_gt_dream)
            
            if self.agent_token_number > 0:
                agent_dream_hidden = dream_hidden[:, self.occ_token_number: self.occ_token_number+self.agent_token_number, :]
                agent_out = self._agent_head_dream(agent_dream_hidden)
                tgt = {"agent_states": agent_states_ft, "agent_labels": agent_labels_ft}
                det_loss = self._agent_loss_v1(tgt, agent_out)
                agent_loss_dream = det_loss[0] + det_loss[1]

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(
                loss_weight, dtype=torch.float32, device=labels.device
            )
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            if ignore_flag:
                loss = loss * 0.0

        if self.agent_token_number > 0:
            loss += agent_loss * 0.1
        if self.gp_token_number > 0:
            loss += gp_loss
        if self.occ_token_number > 0:
            loss += occ_loss
        if self.dream_world:
            loss += occ_loss_dream + agent_loss_dream * 0.1

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    
    def generate_attention_mask(self, input_embeds, input_ids):
        B, N, C = input_embeds.shape
        attention_masks = []
        for i in range(B):
            attention_mask = torch.tril(torch.ones(N, N, device=input_embeds.device))
            batch_selected = torch.argwhere(input_ids == self.world_token_id)
        
            if len(batch_selected) >= (self.occ_token_number + self.agent_token_number + self.gp_token_number ):

                occ_idx = batch_selected[:self.occ_token_number, 0]
                attention_mask[occ_idx[:, None], occ_idx] = 1
                agent_idx = batch_selected[self.occ_token_number:self.occ_token_number+self.agent_token_number, 0]
                attention_mask[agent_idx[:, None], agent_idx] = 1
                goal_idx = batch_selected[self.occ_token_number+self.agent_token_number:self.occ_token_number+self.agent_token_number+self.gp_token_number, 0]
                attention_mask[goal_idx[:, None], goal_idx] = 1
                
                # attention_mask[occ_idx[:, None], agent_idx] = 1

                attention_mask[agent_idx[:, None], occ_idx] = 0
                attention_mask[goal_idx[:, None], occ_idx] = 0
                attention_mask[goal_idx[:, None], agent_idx] = 0

            if self.dream_world:
                batch_selected_dream = torch.argwhere(input_ids == self.dream_token_id)
                occ_idx_dream = batch_selected_dream[:self.occ_token_number, 0]
                attention_mask[occ_idx_dream[:, None], occ_idx_dream] = 1

                agent_idx_dream = batch_selected_dream[self.occ_token_number:self.occ_token_number+self.agent_token_number, 0]
                attention_mask[agent_idx_dream[:, None], agent_idx_dream] = 1

                attention_mask[occ_idx_dream[:, None], agent_idx] = 0
                attention_mask[occ_idx_dream[:, None], goal_idx] = 0
                attention_mask[agent_idx_dream[:, None], occ_idx] = 0
                attention_mask[agent_idx_dream[:, None], occ_idx_dream] = 0
                attention_mask[agent_idx_dream[:, None], goal_idx] = 0
                
            attention_masks.append(attention_mask.unsqueeze(0))

        attention_mask = torch.cat(attention_masks, dim=0)
        return attention_mask.unsqueeze(1)
    
    def cache_forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        agent_states: Optional[torch.Tensor] = None,  # for 3dbbox
        agent_labels: Optional[torch.Tensor] = None,  # for 3dbbox
        trajectory: Optional[torch.Tensor] = None,  # for trajectory
        goalpoint: Optional[torch.Tensor] = None,  # for GoalPoint
        lidar_gt: Optional[torch.Tensor] = None,  # for Occ
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        ego_status: Optional[torch.Tensor] = None, 
        hist_traj: Optional[torch.Tensor] = None, 
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f"dynamic ViT batch size: {vit_batch_size}, dynamic ViT size per sample: {vit_batch_size / B}, dynamic token length: {N}"
            )
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = (
                    statistics.tolist()
                )
                self.num_samples += num_samples
                print(
                    f"total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}"
                )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                f"vit_embeds.shape={vit_embeds.shape}"
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        # replace world token with world query
        if self.world_token_id is not None:
            selected = input_ids == self.world_token_id
            world_embeds = self.world_queries.repeat(B, 1, 1)  #
            world_embeds = self._wm_encoder(vit_embeds, world_embeds)
            world_embeds = self.interact_ego_status(world_embeds, ego_status)
            world_embeds = self.interact_hist_traj(world_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + world_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                world_embeds = world_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"world_embeds.shape={world_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + world_embeds[:n_token]
                )
                ignore_flag = True
        
        if self.dream_token_id is not None:
            selected = input_ids == self.dream_token_id
            dream_embeds = self.dream_queries.repeat(B, 1, 1)  #
            dream_embeds = self._wm_encoder_dream(vit_embeds, dream_embeds)
            dream_embeds = self.interact_ego_status(dream_embeds, ego_status)
            dream_embeds = self.interact_hist_traj(dream_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + dream_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                dream_embeds = dream_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"dream_embeds.shape={dream_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + dream_embeds[:n_token]
                )
                ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        attention_mask = self.cache_generate_attention_mask(input_embeds, input_ids)

        outputs = self.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=True,
            return_dict=return_dict,
        )

        return CausalLMOutputWithPast(
            loss=None,
            logits=None,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
    

    def cache_generate_attention_mask(self, input_embeds, input_ids):
        self.occ_token_number = int(os.getenv("OCC_TOKEN_NUMBER", 625))
        self.agent_token_number = int(os.getenv("AGENT_TOKEN_NUMBER", 50))
        self.gp_token_number = int(os.getenv("GP_TOKEN_NUMBER", 1))
        B, N, C = input_embeds.shape
        attention_masks = []
        for i in range(B):
            attention_mask = torch.tril(torch.ones(N, N, device=input_embeds.device))
            batch_selected = torch.argwhere(input_ids == self.world_token_id)
        
            if len(batch_selected) >= (self.occ_token_number + self.agent_token_number + self.gp_token_number ):

                occ_idx = batch_selected[:self.occ_token_number, 0]
                attention_mask[occ_idx[:, None], occ_idx] = 1
                agent_idx = batch_selected[self.occ_token_number:self.occ_token_number+self.agent_token_number, 0]
                attention_mask[agent_idx[:, None], agent_idx] = 1
                goal_idx = batch_selected[self.occ_token_number+self.agent_token_number:self.occ_token_number+self.agent_token_number+self.gp_token_number, 0]
                attention_mask[goal_idx[:, None], goal_idx] = 1
                
                # attention_mask[occ_idx[:, None], agent_idx] = 1

                attention_mask[agent_idx[:, None], occ_idx] = 0
                attention_mask[goal_idx[:, None], occ_idx] = 0
                attention_mask[goal_idx[:, None], agent_idx] = 0

            if self.dream_world:
                batch_selected_dream = torch.argwhere(input_ids == self.dream_token_id)
                occ_idx_dream = batch_selected_dream[:self.occ_token_number, 0]
                attention_mask[occ_idx_dream[:, None], occ_idx_dream] = 1

                agent_idx_dream = batch_selected_dream[self.occ_token_number:self.occ_token_number+self.agent_token_number, 0]
                attention_mask[agent_idx_dream[:, None], agent_idx_dream] = 1

                attention_mask[occ_idx_dream[:, None], agent_idx] = 0
                attention_mask[occ_idx_dream[:, None], goal_idx] = 0
                attention_mask[agent_idx_dream[:, None], occ_idx] = 0
                attention_mask[agent_idx_dream[:, None], occ_idx_dream] = 0
                attention_mask[agent_idx_dream[:, None], goal_idx] = 0
                
            attention_masks.append(attention_mask.unsqueeze(0))

        attention_mask = torch.cat(attention_masks, dim=0)
        return attention_mask.unsqueeze(1)


    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == "v1":
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                "which results in a transposed image."
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print("Now multi-turn chat is not supported in batch_chat.")
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                "Warning: `image_counts` is deprecated. Please use `num_patches_list` instead."
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and "<image>" not in question:
                question = "<image>\n" + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = "left"
        model_inputs = tokenizer(queries, return_tensors="pt", padding=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config["eos_token_id"] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [
            response.split(template.sep.strip())[0].strip() for response in responses
        ]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        ego_status= None,
        hist_traj=None,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        WORLD_TOKEN='<WORLD TOKEN>',
        WORLD_START_TOKEN="<world>",
        WORLD_END_TOKEN="</world>",
        DREAM_TOKEN="<DREAM TOKEN>",
        DREAM_START_TOKEN="<dream>",
        DREAM_END_TOKEN="</dream>",
        output_text=False,
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "".join(["<image>"] * len(num_patches_list)) + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        
        if self.world_token_id is not None:
            world_cnt = query.count('<world>') # should match the number of world queries
            world_tokens = f"{WORLD_START_TOKEN}{WORLD_TOKEN * world_cnt}{WORLD_END_TOKEN}"
            query = query.replace(
                "<world>" * world_cnt, world_tokens, 1
            )
        
        if self.dream_token_id:
            dream_cnt = query.count(
                "<dream>"
            )  # should match the number of dream queries
            dream_tokens = f"{DREAM_START_TOKEN}{DREAM_TOKEN * dream_cnt}{DREAM_END_TOKEN}"
            query = query.replace(
                "<dream>" * dream_cnt, dream_tokens, 1
            )

        model_inputs = tokenizer(query, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        generation_config["eos_token_id"] = eos_token_id

        # Currently implemented this way: a separate inference executes forward, so generate is not needed here
        if self.world_token_id and output_text:
            text_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                ego_status=ego_status,
                hist_traj=hist_traj,
                **generation_config
            )
            response = tokenizer.batch_decode(text_output, skip_special_tokens=True)[0]
            text = response.split(template.sep.strip())[0].strip()
            generate_out = self.inference(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                ego_status=ego_status, 
                hist_traj=hist_traj, 
            )
            generate_out['text'] = text
            return generate_out
        elif self.world_token_id:
            generate_out = self.inference(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            return generate_out
        else:
            #output text
            generation_output = self.generate(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                **generation_config
            )
            response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[0]
            response = response.split(template.sep.strip())[0].strip()
            history.append((question, response))
            if return_history:
                return response, history
            else:
                query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
                query_to_print = query_to_print.replace(f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>')
                if verbose:
                    print(query_to_print, response)
                return response
            
            
    def chat_onlytext(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        ego_status=None,
        hist_traj=None,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN="<img>",
        IMG_END_TOKEN="</img>",
        IMG_CONTEXT_TOKEN="<IMG_CONTEXT>",
        WORLD_TOKEN='<WORLD TOKEN>',
        WORLD_START_TOKEN="<world>",
        WORLD_END_TOKEN="</world>",
        DREAM_TOKEN="<DREAM TOKEN>",
        DREAM_START_TOKEN="<dream>",
        DREAM_END_TOKEN="</dream>",
        output_text=False,
        verbose=False,
    ):

        if history is None and pixel_values is not None and "<image>" not in question:
            question = "".join(["<image>"] * len(num_patches_list)) + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f"dynamic ViT batch size: {image_bs}")

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace("<image>", image_tokens, 1)
        
        # if self.world_token_id is not None:
        #     world_cnt = query.count('<world>') # default need to modify  should be same with world query numbers
        #     # for i in range(world_cnt):
        #     #     world_tokens = f'{WORLD_TOKEN}'
        #     #     query = query.replace('<world>', world_tokens, 1)
        #     world_tokens = f"{WORLD_START_TOKEN}{WORLD_TOKEN * world_cnt}{WORLD_END_TOKEN}"
        #     query = query.replace(
        #         "<world>" * world_cnt, world_tokens, 1
        #     )
        
        # if self.dream_token_id:
        #     dream_cnt = query.count(
        #         "<dream>"
        #     )  # default need to modify  should be same with world query numbers
        #     dream_tokens = f"{DREAM_START_TOKEN}{DREAM_TOKEN * dream_cnt}{DREAM_END_TOKEN}"
        #     query = query.replace(
        #         "<dream>" * dream_cnt, dream_tokens, 1
        #     )

        model_inputs = tokenizer(query, return_tensors="pt")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        input_ids = model_inputs["input_ids"].to(device)
        attention_mask = model_inputs["attention_mask"].to(device)
        generation_config["eos_token_id"] = eos_token_id

        text_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config
        )
        response = tokenizer.batch_decode(text_output, skip_special_tokens=True)[0]
        text = response.split(template.sep.strip())[0].strip()
        generate_out = {}
        generate_out['text'] = text
        return generate_out

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        ego_status: Optional[torch.Tensor] = None,
        hist_traj: Optional[torch.Tensor] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            # input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
        
        # replace world token with world query
        if self.world_token_id is not None:
            selected = input_ids == self.world_token_id
            world_embeds = self.world_queries.repeat(B, 1, 1)  #
            world_embeds = self._wm_encoder(vit_embeds, world_embeds)
            world_embeds = self.interact_ego_status(world_embeds, ego_status)
            world_embeds = self.interact_hist_traj(world_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + world_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                world_embeds = world_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"world_embeds.shape={world_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + world_embeds[:n_token]
                )
                ignore_flag = True
        
        if self.dream_token_id is not None:
            selected = input_ids == self.dream_token_id
            dream_embeds = self.dream_queries.repeat(B, 1, 1)  #
            dream_embeds = self._wm_encoder_dream(vit_embeds, dream_embeds)
            dream_embeds = self.interact_ego_status(dream_embeds, ego_status)
            dream_embeds = self.interact_hist_traj(dream_embeds, hist_traj)
            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + dream_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                dream_embeds = dream_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"dream_embeds.shape={dream_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + dream_embeds[:n_token]
                )
                ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs
    
    @torch.no_grad()
    def inference(
            self,
            pixel_values: torch.FloatTensor,
            input_ids: torch.LongTensor = None,
            attention_mask: Optional[torch.Tensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            visual_features: Optional[torch.FloatTensor] = None,
            past_key_values: Optional[List[torch.FloatTensor]] = None,
            use_cache: Optional[bool] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            ego_status: Optional[torch.Tensor] = None, 
            hist_traj: Optional[torch.Tensor] = None, 
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.img_context_token_id)
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        # replace world token with world query
        if self.world_token_id is not None:
            selected = (input_ids == self.world_token_id)
            world_embeds = self.world_queries.repeat(B,1,1)
            world_embeds = self._wm_encoder(vit_embeds, world_embeds)
            world_embeds = self.interact_ego_status(world_embeds, ego_status)
            world_embeds = self.interact_hist_traj(world_embeds, hist_traj)

            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)
            try:
                input_embeds[selected] =  input_embeds[selected] * 0.0 + world_embeds.reshape(-1, C)
            except Exception as e:
                world_embeds = world_embeds.reshape(-1, C)
                print(f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                    f'world_embeds.shape={world_embeds.shape}')
                n_token = selected.sum()
                input_embeds[selected] = input_embeds[selected] * 0.0 + world_embeds[:n_token]
        
        if self.dream_token_id is not None:
            selected = input_ids == self.dream_token_id
            dream_embeds = self.dream_queries.repeat(B, 1, 1)  #
            dream_embeds = self._wm_encoder_dream(vit_embeds, dream_embeds)
            dream_embeds = self.interact_ego_status(dream_embeds, ego_status)
            dream_embeds = self.interact_hist_traj(dream_embeds, hist_traj)

            try:
                input_embeds[selected] = input_embeds[
                    selected
                ] * 0.0 + dream_embeds.reshape(-1, C)
                ignore_flag = False
            except Exception as e:
                dream_embeds = dream_embeds.reshape(-1, C)
                print(
                    f"warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, "
                    f"dream_embeds.shape={dream_embeds.shape}"
                )
                n_token = selected.sum()
                input_embeds[selected] = (
                    input_embeds[selected] * 0.0 + dream_embeds[:n_token]
                )
                ignore_flag = True
                
        input_embeds = input_embeds.reshape(B, N, C)

        output_hidden_states=True
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        
        # world_mask = (input_ids == self.world_token_id)
        # world_mask = world_mask.reshape(B,N) # TODO add dream mask

        attention_mask = self.generate_attention_mask(input_embeds, input_ids)

        outputs = self.language_model.forward_with_wm_head(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            mask = None
        )
        # Currently, the output does not consider language, so only the corresponding world tokens are extracted and passed through a separate head
        
        if self.world_token_id is not None:
            selected = (input_ids == self.world_token_id)
            last_hidden_state = outputs.hidden_states[-1].view(-1, C)
            world_hidden = last_hidden_state[selected].view(B,-1,C).contiguous()
            
            outputs={}
            if self.occ_token_number > 0:
                occ_world_hidden = world_hidden[:,: self.occ_token_number, :]
                cls_pred = self._occ_head(occ_world_hidden, [[100, 100], [50, 50]])
                cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2).argmax(1)
                outputs['occ_out'] = cls_pred

            if self.agent_token_number > 0:
                agent_world_hidden = world_hidden[:, self.occ_token_number : self.occ_token_number+self.agent_token_number, :]
                agent_out = self._agent_head(agent_world_hidden)
                outputs['agent_out'] = agent_out

            if self.gp_token_number > 0:
                gp_world_hidden = world_hidden[:,self.occ_token_number+self.agent_token_number : self.occ_token_number+self.agent_token_number+self.gp_token_number, :]
                gp_out = self._gp_head(gp_world_hidden)
                outputs['gp_out'] = gp_out

        if self.dream_token_id is not None:
            selected = (input_ids == self.dream_token_id)
            dream_hidden = last_hidden_state[selected].view(B,-1,C).contiguous()
            
            if self.occ_token_number > 0:
                occ_dream_hidden = dream_hidden[:,: self.occ_token_number, :]
                cls_pred = self._occ_head_dream(occ_dream_hidden, [[100, 100], [50, 50]])
                cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2).argmax(1)
                outputs['occ_out_dream'] = cls_pred

            if self.agent_token_number > 0:
                agent_dream_hidden = dream_hidden[:, self.occ_token_number : self.occ_token_number+self.agent_token_number, :]
                agent_out = self._agent_head_dream(agent_dream_hidden)
                outputs['agent_out_dream'] = agent_out
                
        outputs['logits'] = None
        return outputs

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()

    def _agent_loss(self, targets, predictions):
        """
        Multi-class Hungarian matching loss.
        """
        gt_states, gt_labels = (
            targets["agent_states"],
            targets["agent_labels"],
        )  # gt_labels: (b, n_gt)
        pred_states, pred_logits = (
            predictions["agent_states"],
            predictions["agent_labels"],
        )  # pred_logits: (b, n_pred, C)

        batch_dim, num_pred = pred_states.shape[:2]
        num_gt = gt_labels.shape[1]
        num_classes = pred_logits.shape[-1]

        gt_valid = gt_labels >= 0  # ignore invalid = -1
        num_gt_instances = gt_valid.sum().clamp(min=1)

        # cost matrices
        ce_cost = _get_ce_cost(gt_labels, pred_logits)
        l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)

        cost = 2.0 * ce_cost + 0.5 * l1_cost
        cost = cost.cpu()

        indices = [linear_sum_assignment(c) for c in cost]
        matching = [
            (
                torch.as_tensor(i, dtype=torch.int64),
                torch.as_tensor(j, dtype=torch.int64),
            )
            for i, j in indices
        ]
        idx = _get_src_permutation_idx(matching)

        # matched pairs
        pred_states_idx = pred_states[idx]
        gt_states_idx = torch.cat(
            [t[i] for t, (_, i) in zip(gt_states, indices)], dim=0
        )

        pred_logits_idx = pred_logits[idx]
        gt_labels_idx = torch.cat(
            [t[i] for t, (_, i) in zip(gt_labels, indices)], dim=0
        )

        # losses
        l1_loss = F.l1_loss(pred_states_idx, gt_states_idx, reduction="none").sum(-1)
        l1_loss = l1_loss.view(batch_dim, -1).sum() / num_gt_instances

        ce_loss = F.cross_entropy(
            pred_logits_idx, gt_labels_idx, ignore_index=-1, reduction="mean"
        )

        return ce_loss, l1_loss
    
    def _agent_loss_v1(self, targets, predictions):
        """
        Multi-class Hungarian matching loss with explicit 'no object' class.
        Supports padding (gt_label = -1) and batch-wise matching.
        """

        gt_states, gt_labels = targets["agent_states"], targets["agent_labels"]   # (B, n_gt, 7), (B, n_gt)
        pred_states, pred_logits = predictions["agent_states"], predictions["agent_labels"]  # (B, n_pred, C)

        B, n_pred, num_classes = pred_logits.shape
        n_gt = gt_labels.shape[1]
        assert num_classes >= 2, "Expected at least 2 classes (object + no-object)"

        # Valid GT mask
        gt_valid = (gt_labels >= 0)  # padding = -1
        num_gt_instances = gt_valid.sum().clamp(min=1)

        # === Hungarian Matching ===
        ce_cost = _get_ce_cost_v1(gt_labels, pred_logits[..., :-1])   # Excluding no-object class
        l1_cost = _get_l1_cost(gt_states, pred_states, gt_valid)
        # total_cost = 2.0 * ce_cost + 0.5 * l1_cost #default
        total_cost = 10.0 * ce_cost + 0.5 * l1_cost  # 10.0&0.5
        total_cost = total_cost.cpu()

        indices = []
        for b_idx in range(B):
            valid_gt_mask = gt_valid[b_idx].cpu()
            if valid_gt_mask.sum() == 0:
                indices.append((torch.empty(0, dtype=torch.long),
                                torch.empty(0, dtype=torch.long)))
                continue

            cost_valid = total_cost[b_idx][:, valid_gt_mask]
            pred_ind, gt_ind = linear_sum_assignment(cost_valid)

            # Move to GPU
            indices.append((
                torch.as_tensor(pred_ind, dtype=torch.long, device=pred_logits.device),
                torch.as_tensor(valid_gt_mask.nonzero(as_tuple=False)[gt_ind].squeeze(-1),
                                dtype=torch.long, device=pred_logits.device)
            ))

        # === L1 Regression Loss ===
        l1_loss_total = 0.0
        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(gt_idx) == 0:
                continue
            pred_b = pred_states[b_idx, pred_idx]
            gt_b = gt_states[b_idx, gt_idx].to(pred_states.device)
            l1_loss_total += F.l1_loss(pred_b, gt_b, reduction="none").sum()

        l1_loss = l1_loss_total / num_gt_instances
        l1_loss = l1_loss + (pred_states * 0.0).sum()

        # === Classification CE Loss ===
        # Initialize as no-object class (the last class)
        full_labels = torch.full((B, n_pred), fill_value=num_classes - 1,
                                dtype=torch.long, device=pred_logits.device)

        for b_idx, (pred_idx, gt_idx) in enumerate(indices):
            if len(pred_idx) > 0:
                full_labels[b_idx, pred_idx] = gt_labels[b_idx, gt_idx]
        weight = torch.ones(num_classes, 
                    device=pred_logits.device, 
                    dtype=pred_logits.dtype)
        weight[num_classes - 1] = 0.1
        if (full_labels < 0).any() and torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            raise ValueError("full_labels contains -1 (invalid index). Please check your label preprocessing!")
        ce_loss = F.cross_entropy(
            pred_logits.view(-1, num_classes),
            full_labels.view(-1),
            weight=weight,
            reduction="mean"
        )

        return ce_loss, l1_loss

class AgentHead(nn.Module):
    """Bounding box prediction head."""

    def __init__(
        self,
        num_agents: int,
        d_ffn: int,
        d_model: int,
    ):
        """
        Initializes prediction head.
        :param num_agents: maximum number of agents to predict
        :param d_ffn: dimensionality of feed-forward network
        :param d_model: input dimensionality
        """
        super(AgentHead, self).__init__()

        self._num_objects = num_agents
        self._d_model = d_model
        self._d_ffn = d_ffn
        # self._num_classes = 3 # Original version without no-object category
        self._num_classes = 3 + 1 # Added no-object category

        self._mlp_states = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn // 2),
            nn.ReLU(),
            nn.Linear(self._d_ffn // 2, 7),
        )

        self._mlp_label = nn.Sequential(
            nn.Linear(self._d_model, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn),
            nn.ReLU(),
            nn.Linear(self._d_ffn, self._d_ffn // 8),
            nn.ReLU(),
            nn.Linear(self._d_ffn // 8, self._num_classes),
        )

    def forward(self, agent_queries) -> Dict[str, torch.Tensor]:
        """Torch module forward pass."""

        agent_states = self._mlp_states(agent_queries)
        agent_states[..., 0] = (agent_states[..., 0].tanh() + 1) / 2 * 32
        agent_states[..., 1] = agent_states[..., 1].tanh() * 20
        agent_states[..., 6] = agent_states[..., 6].tanh() * np.pi
        # for name, param in self._mlp_states.named_parameters():
        #     print(f"{name}: mean={param.data.mean():.4f}, std={param.data.std():.4f}, min={param.data.min():.4f}, max={param.data.max():.4f}")

        agent_labels = self._mlp_label(agent_queries).squeeze(dim=-1)

        return {"agent_states": agent_states, "agent_labels": agent_labels}


class GoalPointHead(nn.Module):
    def __init__(
        self,
        language_hidden_dim: int = 896,
        output_dim: int = 2,
    ):
        super(GoalPointHead, self).__init__()
        self.language_hidden_dim = language_hidden_dim
        self.layers = nn.Sequential(
            *linear_relu_ln(language_hidden_dim, 4, 4),
            nn.Linear(language_hidden_dim, output_dim),
        )
        
    def forward(
        self,
        input_feature: torch.Tensor,
    ):
        output = self.layers(input_feature)
        return output
    
    def loss(self, pred_goalpoint, gt_goalpoint):
        loss = F.l1_loss(pred_goalpoint, gt_goalpoint, reduction="mean")
        return loss


class TrajCrossAttentionFusion(nn.Module):
    def __init__(self, d_model=1536, nhead=8):
        super().__init__()
        self.traj_proj = nn.Linear(3, d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, batch_first=True)

    def forward(self, query_feat, hist_traj):
        traj_encoded = self.traj_proj(hist_traj)    # (B, 4, d_model)
        # Let query_feat be Q, and hist_traj be K/V
        fused, _ = self.attn(query_feat, traj_encoded, traj_encoded)
        return fused


class TrajEncoder(nn.Module):
    def __init__(self, d_model=1546):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Flatten(),                # (B, 4, 3) -> (B, 12)
            nn.Linear(12, d_model),      # Project to d_model dimensions
            nn.ReLU(),
            nn.Linear(d_model, d_model)  # Optional second mapping
        )
    def forward(self, traj):
        return self.mlp(traj)


@torch.no_grad()
def _get_ce_cost(gt_labels: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute multi-class cross-entropy cost matrix.

    Args:
        gt_labels: (b, n_gt) int64 tensor with class indices (0..num_classes-1)
        pred_logits: (b, n_pred, num_classes)
    Returns:
        ce_cost: (b, n_pred, n_gt)
    """
    b, n_pred, num_classes = pred_logits.shape
    n_gt = gt_labels.shape[1]
    pred_logits_exp = pred_logits.unsqueeze(2).expand(-1, -1, n_gt, -1)
    gt_labels_exp = gt_labels.unsqueeze(1).expand(-1, n_pred, -1)
    ce_cost = F.cross_entropy(
        pred_logits_exp.reshape(-1, num_classes),
        gt_labels_exp.reshape(-1),
        ignore_index=-1,
        reduction="none",
    ).view(b, n_pred, n_gt)
    return ce_cost


@torch.no_grad()
def _get_l1_cost(
    gt_states: torch.Tensor, pred_states: torch.Tensor, gt_valid: torch.Tensor
) -> torch.Tensor:
    """
    Function to calculate L1 cost for cost matrix (unchanged).
    Only use valid GTs.
    """
    gt_states_expanded = gt_states[:, :, None, :2].detach()  # (b, n_gt, 1, 2)
    pred_states_expanded = pred_states[:, None, :, :2].detach()  # (b, 1, n_pred, 2)
    l1_cost = gt_valid[..., None].float() * (
        gt_states_expanded - pred_states_expanded
    ).abs().sum(-1)
    l1_cost = l1_cost.permute(0, 2, 1)
    return l1_cost


def _get_src_permutation_idx(indices):
    batch_idx = torch.cat(
        [torch.full_like(src, i) for i, (src, _) in enumerate(indices)]
    )
    src_idx = torch.cat([src for (src, _) in indices])
    return batch_idx, src_idx

@torch.no_grad()
def _get_ce_cost_v1(gt_labels: torch.Tensor, pred_logits: torch.Tensor) -> torch.Tensor:
    """
    Compute multi-class cross-entropy cost (excluding 'no-object' class).
    Args:
        gt_labels: (B, n_gt)
        pred_logits: (B, n_pred, C-1)
    Returns:
        ce_cost: (B, n_pred, n_gt)
    """
    B, n_pred, num_classes = pred_logits.shape
    n_gt = gt_labels.shape[1]

    pred_logits_exp = pred_logits.unsqueeze(2).expand(-1, -1, n_gt, -1)
    gt_labels_exp = gt_labels.unsqueeze(1).expand(-1, n_pred, -1)

    ce_cost = F.cross_entropy(
        pred_logits_exp.reshape(-1, num_classes),
        gt_labels_exp.reshape(-1),
        ignore_index=-1,
        reduction='none'
    ).view(B, n_pred, n_gt)

    return ce_cost

def _get_src_permutation_idx_v1(indices, device=None):
    # indices: list of (src_idx, tgt_idx)
    if device is None:
        device = indices[0][0].device if isinstance(indices[0][0], torch.Tensor) else torch.device("cpu")

    batch_idx = torch.cat([
        torch.full_like(src, i, device=device) for i, (src, _) in enumerate(indices)
    ])
    src_idx = torch.cat([src.to(device) for (src, _) in indices])
    return batch_idx, src_idx