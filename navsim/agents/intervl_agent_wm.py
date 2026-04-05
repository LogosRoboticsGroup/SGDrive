import sys
import os
import re
from enum import IntEnum
from typing import Any, Dict, List, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.transforms import InterpolationMode
from lmdeploy import (
    ChatTemplateConfig,
    GenerationConfig,
    PytorchEngineConfig,
    TurbomindEngineConfig,
    pipeline,
)
# from lmdeploy.vl import load_image
from PIL import Image
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers import AutoModel, AutoTokenizer

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import (
    AgentInput,
    Annotations,
    Scene,
    SensorConfig,
    Trajectory,
)
from navsim.common.enums import BoundingBoxIndex, LidarIndex
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling

from internvl_chat.internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig_WM,
                                          InternVLChatModel_WM)
from internvl_chat.internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, QUAD_END_TOKEN,
                                      QUAD_START_TOKEN, REF_END_TOKEN,
                                      REF_START_TOKEN, LOC_START_TOKEN, LOC_END_TOKEN,
                                      FRONT_VIEW_TOKEN, FRONT_LEFT_VIEW_TOKEN, FRONT_RIGHT_VIEW_TOKEN,
                                      BACK_LEFT_VIEW_TOKEN, BACK_RIGHT_VIEW_TOKEN, BACK_VIEW_TOKEN, WORLD_TOKEN, DREAM_TOKEN)

system_message = """
You are a perception model for autonomous driving. Your task is to perceive the scene based on front-view images.
"""

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_input, input_size=448, max_num=12):
    if isinstance(image_input, (str, bytes)):
        image = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype('uint8')).convert('RGB')

    else:
        raise TypeError(f"Unsupported image input type: {type(image_input)}")
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def format_number(n, decimal_places=2):
    if abs(round(n, decimal_places)) <= 1e-2:
        return 0.0
    else:
        format_string = f"{{n:+.{decimal_places}f}}"
        return format_string.format(n=n)

class InterVLPredictor(nn.Module):
    def __init__(self, model_path, is_train=False):
        super().__init__()
        self.model, self.tokenizer = self.load_model(model_path, is_train)

    def load_model(self, model_path, is_train):
        config = InternVLChatConfig_WM.from_pretrained(model_path)
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

        model = InternVLChatModel_WM.from_pretrained(
            model_path,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=None   
        )

        if not is_train:
            model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True,
            use_fast=False
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)   
        world_token_id = tokenizer.convert_tokens_to_ids(WORLD_TOKEN)
        model.world_token_id = world_token_id   

        dream_token_id = tokenizer.convert_tokens_to_ids(DREAM_TOKEN)
        model.dream_token_id = dream_token_id           

        print("****** Finished initializing Model and Tokenizer *****")
        return model, tokenizer

class InternVLFeatureBuilder(AbstractFeatureBuilder):

    def __init__(self):
        """Initializes the feature builder."""
        pass

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras
        return {"ego_statuses": ego_statuses,"cameras": cameras}


class TrajectoryTargetBuilder(AbstractTargetBuilder):
    """Input feature builder of EgoStatusMLP."""

    def __init__(self, trajectory_sampling: TrajectorySampling):
        """
        Initializes the target builder.
        :param trajectory_sampling: trajectory sampling specification.
        """

        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        """Inherited, see superclass."""
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        future_trajectory = scene.get_future_trajectory(num_trajectory_frames=self._trajectory_sampling.num_poses)

        return {
            "trajectory": torch.tensor(future_trajectory.poses),
        }
    
class InternVLAgent_WM(AbstractAgent):

    def __init__(
        self,
        trajectory_sampling: TrajectorySampling,
        checkpoint_path: Optional[str] = None,
        prompt_type: Optional[str] = "base",
        cam_type: Optional[str] = "single",
    ):
        """Initializes the InternVLAgent.

        Args:
            trajectory_sampling (TrajectorySampling): The specification for sampling future trajectories.
            checkpoint_path (Optional[str]): Path to the model checkpoint to be loaded. Defaults to None.
        """
        super().__init__()
        self._trajectory_sampling = trajectory_sampling
        self.checkpoint_path = checkpoint_path
        self.predictor =  InterVLPredictor(model_path = self.checkpoint_path, is_train = False)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.predictor.model.to(self.device)
        self.prompt_type = prompt_type
        self.cam_type = cam_type     


    def name(self) -> str:
        """Inherited, see superclass."""
        return self.__class__.__name__

    def initialize(self) -> None:
        """Inherited, see superclass."""
        pass

    def get_sensor_config(self) -> SensorConfig:
        """Inherited, see superclass."""
        return SensorConfig.build_all_sensors(include=[0, 1, 2, 3])

    def get_target_builders(self) -> List[AbstractTargetBuilder]:
        """Inherited, see superclass."""
        return [TrajectoryTargetBuilder(trajectory_sampling=self._trajectory_sampling)]

    def get_feature_builders(self) -> List[AbstractFeatureBuilder]:
        """Inherited, see superclass."""
        return [InternVLFeatureBuilder()]
    
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


    def forward(self, features: Dict[str, torch.Tensor],targets=None, output_text=False) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        cameras = features["cameras"]
        ego_statuses = features["ego_statuses"]

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

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ["turn left", "go straight", "turn right"]
        command_str = [
            navigation_commands[i]
            for i in range(len(high_command_one_hot))
            if high_command_one_hot[i] == 1
        ]
        command_str = command_str[0] if command_str else "unknown"

        # image_paths = []
        image_inputs = []
        image_prompt_lines = []
        for i in range(4):
            # image_paths.append(str(cameras[i].cam_f0.image)) #eval
            image_inputs.append(cameras[i].cam_f0.image)  
            image_prompt_lines.append("<image>")
            # print(image_paths)
            
            # image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n")
        # image_prompt = "1. Visual perception from front camera view (last 4 timesteps)\n"
        image_prompt_desc = "1. Visual perception from front camera view (last 4 timesteps)\n"

        # pixel_values = [load_image(image_path) for image_path in image_paths]
        pixel_values_list = []
        if type(image_inputs) == list:
            for image_input in image_inputs:
                pixel_values = load_image(image_input, max_num=4).to(torch.bfloat16).to(self.device)
                pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        generation_config = dict(max_new_tokens=512, min_new_tokens=50, do_sample=False, num_beams=1, pad_token_id=self.predictor.tokenizer.eos_token_id)

        image_prompt_str = "".join(image_prompt_lines)

        # common_prompt = f"""As an autonomous driving system, preception the current scene based on:\n{image_prompt}"""
        common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        if self.prompt_type == "vel_and_acc":
            current_ego_status = ego_statuses[-1]
            vel_acc_info = (
                f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, {format_number(current_ego_status.ego_velocity[1])})"
                f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, {format_number(current_ego_status.ego_acceleration[1])})"
            )
            question = f"{image_prompt_str}\n{common_prompt}{vel_acc_info}{output_requirements}"
        else:
            question = f"{''.join([f'<image>' for i in range(len(image_inputs))])}\n{common_prompt}{output_requirements}"

        question += "\n" + "".join(["<world>"] * self.predictor.model.NUM_WORLD_TOKEN)

        if self.predictor.model.dream_world:
            question += "\n" + "".join(["<dream>"] * (self.predictor.model.occ_token_number + self.predictor.model.agent_token_number))
        
        ego_status, hist_traj = self.preprocess_ego_status_hist_traj(features["ego_statuses"])
        if output_text:
            responses = self.predictor.model.chat(
                self.predictor.tokenizer,
                pixel_values,
                question,
                generation_config,
                ego_status=ego_status, 
                hist_traj=hist_traj,
                num_patches_list=num_patches_list,
                output_text=True
            )
            return responses
            # bboxes = responses["agent_states"][0]
            # labels = responses["agent_labels"][0]
            # text = responses["text"]
            # return bboxes, labels, text
        else:
            responses = self.predictor.model.chat(
                self.predictor.tokenizer,
                pixel_values,
                question,
                generation_config,
                ego_status=ego_status, 
                hist_traj=hist_traj, 
                num_patches_list=num_patches_list,
                output_text=False
            )
            return responses
        #     bboxes = responses["agent_states"][0]
        #     labels = responses["agent_labels"][0]
        
        # return bboxes, labels, None

    def forward_traj(self, features: Dict[str, torch.Tensor],targets=None, output_text=False) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        cameras = features["cameras"]
        ego_statuses = features["ego_statuses"]

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

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ["turn left", "go straight", "turn right"]
        command_str = [
            navigation_commands[i]
            for i in range(len(high_command_one_hot))
            if high_command_one_hot[i] == 1
        ]
        command_str = command_str[0] if command_str else "unknown"

        image_paths = []
        # image_inputs = []
        image_prompt_lines = []
        for i in range(4):
            image_paths.append(str(cameras[i].cam_f0.image)) #eval
            # image_inputs.append(cameras[i].cam_f0.image)  
            image_prompt_lines.append("<image>")
            # print(image_paths)
            
            # image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n")
        # image_prompt = "1. Visual perception from front camera view (last 4 timesteps)\n"
        image_prompt_desc = "1. Visual perception from front camera view (last 4 timesteps)\n"

        # pixel_values = [load_image(image_path) for image_path in image_paths]
        pixel_values_list = []
        if type(image_paths) == list:
            for image_path in image_paths:
                # print(image_path)
                pixel_values = load_image(image_path, max_num=4).to(torch.bfloat16).to(self.device)
                pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        generation_config = dict(max_new_tokens=512, min_new_tokens=50, do_sample=False, num_beams=1, pad_token_id=self.predictor.tokenizer.eos_token_id)

        image_prompt_str = "".join(image_prompt_lines)

        # common_prompt = f"""As an autonomous driving system, preception the current scene based on:\n{image_prompt}"""
        common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        if self.prompt_type == "vel_and_acc":
            current_ego_status = ego_statuses[-1]
            vel_acc_info = (
                f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, {format_number(current_ego_status.ego_velocity[1])})"
                f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, {format_number(current_ego_status.ego_acceleration[1])})"
            )
            question = f"{image_prompt_str}\n{common_prompt}{vel_acc_info}{output_requirements}"
        else:
            question = f"{''.join([f'<image>' for i in range(len(image_paths))])}\n{common_prompt}{output_requirements}"

        question += "\n" + "".join(["<world>"] * self.predictor.model.NUM_WORLD_TOKEN)

        if self.predictor.model.dream_world:
            question += "\n" + "".join(["<dream>"] * (self.predictor.model.occ_token_number + self.predictor.model.agent_token_number))
        
        ego_status, hist_traj = self.preprocess_ego_status_hist_traj(features["ego_statuses"])
        # responses = self.predictor.model.chat(
        #     self.predictor.tokenizer,
        #     pixel_values,
        #     question,
        #     generation_config,
        #     ego_status=None, 
        #     hist_traj=None, 
        #     # ego_status=ego_status, 
        #     # hist_traj=hist_traj, 
        #     num_patches_list=num_patches_list,
        # )

        # bboxes = responses["agent_states"][0]
        # labels = responses["agent_labels"][0]
        # print(bboxes.shape)
        # print(labels.shape)
        if output_text:
            responses = self.predictor.model.chat(
                self.predictor.tokenizer,
                pixel_values,
                question,
                generation_config,
                ego_status=ego_status, 
                hist_traj=hist_traj,
                num_patches_list=num_patches_list,
                output_text=True
            )
            return responses
            # bboxes = responses["agent_states"][0]
            # labels = responses["agent_labels"][0]
            # text = responses["text"]
            # return bboxes, labels, text
        else:
            responses = self.predictor.model.chat(
                self.predictor.tokenizer,
                pixel_values,
                question,
                generation_config,
                ego_status=ego_status, 
                hist_traj=hist_traj, 
                num_patches_list=num_patches_list,
                output_text=False
            )
            return responses
        #     bboxes = responses["agent_states"][0]
        #     labels = responses["agent_labels"][0]
        
        # return bboxes, labels, None
    
    def forward_traj_onlytext(self, features: Dict[str, torch.Tensor],targets=None, output_text=False) -> Dict[str, torch.Tensor]:
        """Inherited, see superclass."""
        
        cameras = features["cameras"]
        ego_statuses = features["ego_statuses"]

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

        high_command_one_hot = ego_statuses[-1].driving_command
        navigation_commands = ["turn left", "go straight", "turn right"]
        command_str = [
            navigation_commands[i]
            for i in range(len(high_command_one_hot))
            if high_command_one_hot[i] == 1
        ]
        command_str = command_str[0] if command_str else "unknown"

        image_paths = []
        # image_inputs = []
        image_prompt_lines = []
        for i in range(4):
            image_paths.append(str(cameras[i].cam_f0.image)) #eval
            # image_inputs.append(cameras[i].cam_f0.image)  
            image_prompt_lines.append("<image>")
            # print(image_paths)
            
            # image_prompt_lines.append(f"<FRONT VIEW>Frame-{i+1}: <image>\n")
        # image_prompt = "1. Visual perception from front camera view (last 4 timesteps)\n"
        image_prompt_desc = "1. Visual perception from front camera view (last 4 timesteps)\n"

        # pixel_values = [load_image(image_path) for image_path in image_paths]
        pixel_values_list = []
        if type(image_paths) == list:
            for image_path in image_paths:
                # print(image_path)
                pixel_values = load_image(image_path, max_num=4).to(torch.bfloat16).to(self.device)
                pixel_values_list.append(pixel_values)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        num_patches_list = [pv.size(0) for pv in pixel_values_list]

        generation_config = dict(max_new_tokens=512, min_new_tokens=50, do_sample=False, num_beams=1, pad_token_id=self.predictor.tokenizer.eos_token_id)

        image_prompt_str = "".join(image_prompt_lines)

        # common_prompt = f"""As an autonomous driving system, preception the current scene based on:\n{image_prompt}"""
        common_prompt = f"""As an autonomous driving system, predict the vehicle's trajectory based on:\n{image_prompt_desc}2. Historical motion context (last 4 timesteps):{" ".join([f'   - t-{3-i}: ({t["x"]}, {t["y"]}, {t["heading"]})' for i, t in enumerate(history_trajectory)])}\n3. Active navigation command: [{command_str.upper()}]"""
        output_requirements = (
            "\nOutput requirements:\n- Predict 8 future trajectory points\n"
            "- Each point format: (x:float, y:float, heading:float)\n"
            "- Use [PT, ...] to encapsulate the trajectory\n"
            "- Maintain numerical precision to 2 decimal places"
        )
        if self.prompt_type == "vel_and_acc":
            current_ego_status = ego_statuses[-1]
            vel_acc_info = (
                f"\n4. Current velocity: ({format_number(current_ego_status.ego_velocity[0])}, {format_number(current_ego_status.ego_velocity[1])})"
                f"\n5. Current acceleration: ({format_number(current_ego_status.ego_acceleration[0])}, {format_number(current_ego_status.ego_acceleration[1])})"
            )
            question = f"{image_prompt_str}\n{common_prompt}{vel_acc_info}{output_requirements}"
        else:
            question = f"{''.join([f'<image>' for i in range(len(image_paths))])}\n{common_prompt}{output_requirements}"

        # question += "\n" + "".join(["<world>"] * self.predictor.model.NUM_WORLD_TOKEN)

        # if self.predictor.model.dream_world:
        #     question += "\n" + "".join(["<dream>"] * (self.predictor.model.occ_token_number + self.predictor.model.agent_token_number))
        
        ego_status, hist_traj = self.preprocess_ego_status_hist_traj(features["ego_statuses"]) 
        # responses = self.predictor.model.chat(
        #     self.predictor.tokenizer,
        #     pixel_values,
        #     question,
        #     generation_config,
        #     ego_status=None, 
        #     hist_traj=None, 
        #     # ego_status=ego_status, 
        #     # hist_traj=hist_traj, 
        #     num_patches_list=num_patches_list,
        # )

        # bboxes = responses["agent_states"][0]
        # labels = responses["agent_labels"][0]
        # print(bboxes.shape)
        # print(labels.shape)
        
        responses = self.predictor.model.chat_onlytext(
            self.predictor.tokenizer,
            pixel_values,
            question,
            generation_config,
            ego_status=ego_status, 
            hist_traj=hist_traj,
            num_patches_list=num_patches_list,
            output_text=True
        )
        return responses
            # bboxes = responses["agent_states"][0]
            # labels = responses["agent_labels"][0]
            # text = responses["text"]
            # return bboxes, labels, text
        
    

    def compute_trajectory(self, agent_input: AgentInput) -> Trajectory:
        """
        Computes the ego vehicle trajectory.
        :param current_input: Dataclass with agent inputs.
        :return: Trajectory representing the predicted ego's position in future
        """
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        # with torch.no_grad():
        #     predictions = self.forward(features)
        #     poses = predictions["trajectory"].squeeze(0)


        with torch.no_grad():
            responses = self.forward_traj(features, output_text=True)
            # responses = self.forward_traj_onlytext(features, output_text=True) 
            # print(responses.keys())
            # print(responses['text'])
            answers = [responses["text"]]
            full_match = re.search(
                r"\[PT(?:, )?((?:\([-+]?\d*\.\d+, [-+]?\d*\.\d+, [-+]?\d*\.\d+\)(?:, )?){8})\]",
                answers[0],
            )
            if full_match:
                coords_matches = re.findall(
                    r"\(([-+]?\d*\.\d+), ([-+]?\d*\.\d+), ([-+]?\d*\.\d+)\)",
                    full_match.group(1),
                )
                if len(coords_matches) == 8:
                    coordinates = [tuple(map(float, coord)) for coord in coords_matches]
                    coordinates_array = np.array(coordinates, dtype=np.float32)
                    ans = {
                        "trajectory": coordinates_array.reshape(
                            1, self._trajectory_sampling.num_poses, 3
                        )
                    }
            else:
                print("Error parsing trajectory, returning zeros:", answers)
                ans = {
                    "trajectory": np.zeros(
                        (1, self._trajectory_sampling.num_poses, 3), dtype=np.float32
                    )
                }
            poses = ans["trajectory"].squeeze(0)

        return Trajectory(poses)
    
    def compute_res(self, agent_input: AgentInput, output_text=False) -> Annotations:
        self.eval()
        features: Dict[str, torch.Tensor] = {}

        for builder in self.get_feature_builders():
            features.update(builder.compute_features(agent_input))

        with torch.no_grad():
            output = self.forward(features, output_text=output_text)
            res = {}
            if "agent_out" in output:
                boxes = output["agent_out"]["agent_states"][0]
                probs = output["agent_out"]["agent_labels"][0]
                probs = torch.softmax(probs, dim=-1)
               
                threshold = 0.1
                class_names = ["vehicle", "pedestrian", "bicycle", "no-object"]
                id2class = {i: name for i, name in enumerate(class_names)}
                max_probs, pred_labels_single = probs.max(dim=-1)  
                
                mask = pred_labels_single!=3
                max_probs = max_probs[mask]
                pred_labels_single = pred_labels_single[mask]
                boxes = boxes[mask]
                N = boxes.shape[0]

                names = [
                    id2class[label.item()] if prob >= threshold else "ignore"
                    for label, prob in zip(pred_labels_single, max_probs)
                ]
                print(names)

                velocity_3d = np.zeros((N, 3), dtype=np.float32)  
                instance_tokens = ["" for _ in range(N)]
                track_tokens = ["" for _ in range(N)]

                ann = Annotations(
                    boxes=boxes.to(torch.float32).cpu().numpy(),
                    names=np.array(names),
                    velocity_3d=velocity_3d,
                    instance_tokens=instance_tokens,
                    track_tokens=track_tokens,
                )
                res.update({
                    "boxes": ann
                })
            
            if "agent_out_dream" in output:
                boxes = output["agent_out_dream"]["agent_states"][0]
                probs = output["agent_out_dream"]["agent_labels"][0]
                probs = torch.softmax(probs, dim=-1)
                
                threshold = 0.1
                class_names = ["vehicle", "pedestrian", "bicycle", "no-object"]
                id2class = {i: name for i, name in enumerate(class_names)}
                max_probs, pred_labels_single = probs.max(dim=-1)  
                
                mask = pred_labels_single!=3
                max_probs = max_probs[mask]
                pred_labels_single = pred_labels_single[mask]
                boxes = boxes[mask]
                N = boxes.shape[0]

                names = [
                    id2class[label.item()] if prob >= threshold else "ignore"
                    for label, prob in zip(pred_labels_single, max_probs)
                ]
                print(names)

                velocity_3d = np.zeros((N, 3), dtype=np.float32)   
                instance_tokens = ["" for _ in range(N)]
                track_tokens = ["" for _ in range(N)]

                ann = Annotations(
                    boxes=boxes.to(torch.float32).cpu().numpy(),
                    names=np.array(names),
                    velocity_3d=velocity_3d,
                    instance_tokens=instance_tokens,
                    track_tokens=track_tokens,
                )
                res.update({
                    "boxes_dream": ann
                })

            if "gp_out" in output:
                res.update({
                    "gp": output["gp_out"].to(torch.float32).cpu().numpy()
                })

            if "occ_out" in output:
                res.update({
                    "occ": output["occ_out"].to(torch.float32).cpu().numpy()
                })
            
            if "occ_out_dream" in output:
                res.update({
                    "occ_dream": output["occ_out_dream"].to(torch.float32).cpu().numpy()
                })
            if "text" in output:
                res.update({"text": output["text"]})

        return res

    def compute_loss(
        self, features: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor], predictions: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Inherited, see superclass."""
        return torch.nn.functional.l1_loss(predictions["trajectory"], targets["trajectory"])

    def get_optimizers(self) -> Union[Optimizer, Dict[str, Union[Optimizer, LRScheduler]]]:
        """Inherited, see superclass."""
        return torch.optim.Adam(self._mlp.parameters(), lr=self._lr)
