import os
from typing import List, Optional, Tuple, Union
import torch
from torch import nn
from transformers import AutoModel, AutoTokenizer
from transformers.modeling_outputs import CausalLMOutputWithPast
from internvl_chat.internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig_WM,
                                          InternVLChatModel_WM)


from .utils.conversation import get_conv_template
from ..intervl_agent_wm import format_number

IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"

WORLD_TOKEN = "<WORLD TOKEN>"
WORLD_START_TOKEN = "<world>"
WORLD_END_TOKEN = "</world>"

DREAM_TOKEN = "<DREAM TOKEN>"
DREAM_START_TOKEN = "<dream>"
DREAM_END_TOKEN = "</dream>"

system_message = """
You are a vehicle trajectory prediction model for autonomous driving. Your task is to predict the ego vehicle's 4-second trajectory based on the following inputs: multi-view images from 8 cameras, ego vehicle states (position), and discrete navigation commands. The input provides a 2-second history, and your output should ensure a safe trajectory for the next 4 seconds. Your predictions must adhere to the following metrics:
1. **No at-fault Collisions (NC)**: Avoid collisions with other objects/vehicles.
2. **Drivable Area Compliance (DAC)**: Stay within the drivable area.
3. **Time to Collision (TTC)**: Maintain a safe distance from other vehicles.
4. **Ego Progress (EP)**: Ensure the ego vehicle moves forward without being stuck.
5. **Comfort (C)**: Avoid sharp turns and sudden decelerations.
6. **Driving Direction Compliance (DDC)**: Align with the intended driving direction.
For evaluation, use the **PDM Score**, which combines these metrics: **PDM Score** = NC * DAC * (5*TTC + 5*EP + 2*C + 0*DDC) / 12.
Your predictions will be evaluated through a non-reactive 4-second simulation with an LQR controller and background actors following their recorded trajectories. The better your predictions, the higher your score.
"""


class SGDriveBackbone(nn.Module):
    """
    A simplified vision-language model backbone with direct loading logic
    for different model architectures (InternVL, Qwen-VL).
    """

    def __init__(self, model_type: str, checkpoint_path: str, device: str = "cuda"):
        """
        Initializes and loads the specified model and its preprocessor/tokenizer.

        Args:
            model_type (str): The type of model to load. Supported: 'internvl', 'qwen'.
            checkpoint_path (str): The path to the model checkpoint.
            device (str): The device to load the model onto ('cuda', 'cpu').
        """
        super().__init__()

        self.model = None
        self.tokenizer = None
        self.model_type = model_type.lower()
        self.device = device

        print(
            f"Initializing backbone of type: '{self.model_type}' from path: '{checkpoint_path}'"
        )

        if self.model_type == "internvl":
            # --- Load InternVL Model and Tokenizer ---
            self.model = AutoModel.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.device,
            ).eval()
            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path, trust_remote_code=True, use_fast=False
            )
            # Load model-specific configuration
            self._configure_internvl()
            self.num_image_token = 256

        elif self.model_type == "qwen":
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device,
                trust_remote_code=True,
            )
            self.tokenizer = AutoProcessor.from_pretrained(
                checkpoint_path, trust_remote_code=True
            )
        elif self.model_type == "internvl_wm":
            config = InternVLChatConfig_WM.from_pretrained(checkpoint_path)
            config.output_hidden_states = True
            config.llm_config._attn_implementation = "sdpa"
            config.llm_config._attn_implementation_internal = "sdpa"

            self.num_image_token = 256
            if os.getenv("use_world_token", True):
                self.occ_token_number = int(os.getenv("OCC_TOKEN_NUMBER", 625))
                config.occ_token_number = self.occ_token_number
                self.agent_token_number = int(os.getenv("AGENT_TOKEN_NUMBER", 50))
                config.agent_token_number = self.agent_token_number
                self.gp_token_number = int(os.getenv("GP_TOKEN_NUMBER", 1))
                config.gp_token_number = self.gp_token_number
                self.world_token_number = self.occ_token_number + self.agent_token_number + self.gp_token_number
            if os.getenv("dream_world", True):
                self.dream_token_number = self.occ_token_number + self.agent_token_number
            
            self.model = InternVLChatModel_WM.from_pretrained(
                checkpoint_path,
                config=config,
                torch_dtype=torch.bfloat16,   
                device_map=self.device
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_path,
                trust_remote_code=True,
            )

            self._configure_internvl()
            if os.getenv("use_world_token", True):
                world_token_id = self.tokenizer.convert_tokens_to_ids(WORLD_TOKEN)
                self.model.world_token_id = world_token_id
                self.world_token_id = world_token_id
            if os.getenv("dream_world", True):
                dream_token_id = self.tokenizer.convert_tokens_to_ids(DREAM_TOKEN)
                self.model.dream_token_id = dream_token_id
                self.dream_token_id = dream_token_id
        else:
            raise ValueError(
                f"Unsupported model_type: '{self.model_type}'. Please choose 'internvl' or 'qwen'."
            )

        print(
            f"Backbone '{self.model_type}' loaded successfully on device '{self.device}'."
        )

    def _configure_internvl(self):
        """Applies specific configurations required for the InternVL model."""
        self.model.system_message = system_message
        self.img_context_token_id = self.tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.model.img_context_token_id = self.img_context_token_id
        print("InternVL model configured.")

    def forward(
        self,
        pixel_values: torch.Tensor,
        questions: List[str],
        num_patches_list: List[int],
        ego_status= None,
    ):
        if self.model_type != "internvl_wm":
            if not self.model:
                raise RuntimeError(
                    "Backbone model has not been initialized. Call initialize() on the agent first."
                )

            queries = []
            for idx, num_patches in enumerate(num_patches_list):
                question = questions[idx]
                if pixel_values is not None and "<image>" not in question:
                    question = "<image>\n" + question

                template = get_conv_template("internvl2_5")
                template.system_message = system_message
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
            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(
                queries, return_tensors="pt", padding="max_length", max_length=2800
            )
            device = torch.device("cuda")
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            num_patches = pixel_values.size(0)
            image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

            return self.model(
                pixel_values=pixel_values.bfloat16(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                return_dict=True,
            )
        else:
            if not self.model:
                raise RuntimeError(
                    "Backbone model has not been initialized. Call initialize() on the agent first."
                )

            question = questions[0] + "\n<world>" + "\n<dream>"
            template = get_conv_template("internvl2_5")
            template.system_message = system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            for idx, num_patches in enumerate(num_patches_list):
                # if pixel_values is not None and "<image>" not in question:
                #     question = "<image>\n" + question
                image_tokens = (
                    IMG_START_TOKEN
                    + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                    + IMG_END_TOKEN
                )
                query = query.replace("<image>", image_tokens, 1)

            if self.world_token_number > 0:
                world_tokens = (
                    WORLD_START_TOKEN
                    + WORLD_TOKEN * self.world_token_number
                    + WORLD_END_TOKEN
                )
                query = query.replace("<world>", world_tokens, 1)
            
            if self.dream_token_number > 0:
                dream_tokens = (
                    DREAM_START_TOKEN
                    + DREAM_TOKEN * self.dream_token_number
                    + DREAM_END_TOKEN
                )
                query = query.replace("<dream>", dream_tokens, 1)

            self.tokenizer.padding_side = "left"
            model_inputs = self.tokenizer(query, return_tensors="pt", padding="max_length", max_length=5100) #dream+world
            # model_inputs = self.tokenizer(query, return_tensors="pt", padding="max_length", max_length=4300)
            # model_inputs = self.tokenizer(query, return_tensors="pt", padding="max_length", max_length=(3624 + self.world_token_number))
            device = torch.device("cuda")
            input_ids = model_inputs["input_ids"].to(device)
            attention_mask = model_inputs["attention_mask"].to(device)

            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)

            num_patches = pixel_values.size(0)
            image_flags = torch.tensor([1] * num_patches, dtype=torch.long)

            ego_status, hist_traj = self.preprocess_ego_status_hist_traj(ego_status)

            outputs = self.model.cache_forward(
                pixel_values=pixel_values.bfloat16(),
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                image_flags=image_flags.squeeze(-1),
                output_hidden_states=True,
                ego_status = ego_status,
                hist_traj=hist_traj,
                return_dict=True,
            )
            last_hidden_state = outputs.hidden_states[-1]
            B,N,C = last_hidden_state.shape
            input_ids = input_ids.reshape(B * N)
            final_outputs={}
            if self.world_token_id is not None:
                selected = (input_ids == self.world_token_id)
                last_hidden_state = outputs.hidden_states[-1].view(-1, C)
                world_hidden = last_hidden_state[selected].view(B,-1,C).contiguous()
                
                if self.occ_token_number > 0:
                    occ_world_hidden = world_hidden[:,: self.occ_token_number, :]
                    # cls_pred = self._occ_head(occ_world_hidden, [[100, 100], [50, 50]])
                    # cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2).argmax(1)
                    final_outputs['occ_out'] = occ_world_hidden

                if self.agent_token_number > 0:
                    agent_world_hidden = world_hidden[:, self.occ_token_number : self.occ_token_number+self.agent_token_number, :]
                    # agent_out = self._agent_head(agent_world_hidden)
                    final_outputs['agent_out'] = agent_world_hidden

                if self.gp_token_number > 0:
                    gp_world_hidden = world_hidden[:,self.occ_token_number+self.agent_token_number : self.occ_token_number+self.agent_token_number+self.gp_token_number, :]
                    # gp_out = self._gp_head(gp_world_hidden)
                    final_outputs['gp_out'] = gp_world_hidden

            if self.dream_token_id is not None:
                selected = input_ids == self.dream_token_id
                last_hidden_state = outputs.hidden_states[-1].view(-1, C)
                dream_hidden = last_hidden_state[selected].view(B, -1, C).contiguous()
                
                if self.occ_token_number > 0:
                    occ_dream_hidden = dream_hidden[:, : self.occ_token_number, :]
                    final_outputs['dream_occ_out'] = occ_dream_hidden
                    # cls_pred = self._occ_head_dream(occ_dream_hidden, [[100, 100], [50, 50]])
                    # cls_pred = cls_pred.reshape(B, 2, 16, 100, 100).permute(0,1,3,4,2)
                    # occ_loss_dream = self._occ_head.loss(cls_pred, lidar_gt_dream)
                    # cls_pred.detach().to(torch.float32).cpu().numpy().tofile("/lpai/output/data/cls_pred_dream.npy")
                    # lidar_gt_dream.cpu().numpy().tofile("/lpai/output/data/gt_dream.npy")
                
                if self.agent_token_number > 0:
                    agent_dream_hidden = dream_hidden[:, self.occ_token_number: self.occ_token_number+self.agent_token_number, :]
                    final_outputs['dream_agent_out'] = agent_dream_hidden
                    # agent_out = self._agent_head_dream(agent_dream_hidden)
                    # tgt = {"agent_states": agent_states_ft, "agent_labels": agent_labels_ft}
                    # det_loss = self._agent_loss_v1(tgt, agent_out)
                    # agent_loss_dream = det_loss[0] + det_loss[1]
                    
            final_outputs['outputs'] = outputs
            
            return final_outputs
    
    def preprocess_ego_status_hist_traj(self, ego_statuses):
        driving_command = torch.tensor(ego_statuses[-1].driving_command)
        ego_acceleration = torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32)
        ego_pos = torch.tensor(ego_statuses[-1].ego_pose, dtype=torch.float32)
        ego_velocity = torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32)
        ego_status_ = torch.cat([driving_command, ego_acceleration, ego_pos, ego_velocity]).unsqueeze(0)

        history_trajectory = []
        for i in range(4):
            ego_status = ego_statuses[i]
            history_trajectory.append(
                {
                    "x": format_number(ego_status.ego_pose[0]),
                    "y": format_number(ego_status.ego_pose[1]),
                    "heading": format_number(ego_status.ego_pose[2]),
                }
            )

        hist_traj = torch.tensor([
            [float(d['x']), float(d['y']), float(d['heading'])]
            for d in history_trajectory
        ], dtype=torch.float32)
        return ego_status_, hist_traj.unsqueeze(0)