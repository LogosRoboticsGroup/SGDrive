from typing import Dict, Optional
import torch
import numpy as np
import gzip
import pickle
from PIL import Image

from navsim.agents.abstract_agent import AgentInput
from navsim.planning.training.abstract_feature_target_builder import (
    AbstractFeatureBuilder,
    AbstractTargetBuilder,
)
from navsim.common.dataclasses import Scene, Trajectory
from nuplan.planning.simulation.trajectory.trajectory_sampling import TrajectorySampling
from .sgdrive_backbone import SGDriveBackbone
from .utils.internvl_preprocess import load_image


def format_number(n, decimal_places=2):
    return f"{n:+.{decimal_places}f}" if abs(round(n, decimal_places)) > 1e-2 else "0.0"


class SGDriveFeatureBuilder(AbstractFeatureBuilder):
    def __init__(
        self,
        cache_hidden_state: bool = True,
        model_type: Optional[str] = None,
        checkpoint_path: Optional[str] = None,
        device: str = "cuda",
        cache_mode: bool = False,
    ):
        """
        Initializes the feature builder.

        Args:
            cache_hidden_state (bool): If True, operates in online mode, initializes the backbone,
                                       and computes the hidden state. If False, operates in offline
                                       mode, does not initialize the backbone, and returns
                                       pre-computable tensors, including a tensorized representation
                                       of the image file path.
            model_type (str, optional): The type of model to load ('internvl' or 'qwen'). Required if cache_hidden_state is True.
            checkpoint_path (str, optional): Path to the model checkpoint. Required if cache_hidden_state is True.
            device (str): The device to load the model onto.
        """
        super().__init__()
        self.cache_hidden_state = cache_hidden_state
        self.backbone = None
        self.cache_mode = cache_mode

        if self.cache_hidden_state and self.cache_mode:
            if not model_type or not checkpoint_path:
                raise ValueError(
                    "In online mode (cache_hidden_state=True), `model_type` and `checkpoint_path` must be provided."
                )
            self.backbone = SGDriveBackbone(
                model_type=model_type, checkpoint_path=checkpoint_path, device=device
            )

    def get_unique_name(self) -> str:
        return "internvl_feature"

    def compute_features(self, agent_input: AgentInput) -> Dict[str, torch.Tensor]:

        ego_statuses = agent_input.ego_statuses
        cameras = agent_input.cameras

        history_trajectory = torch.tensor(
            [
                [float(e.ego_pose[0]), float(e.ego_pose[1]), float(e.ego_pose[2])]
                for e in ego_statuses[:4]
            ],
            dtype=torch.float32,
        )
        high_command_one_hot = torch.tensor(
            ego_statuses[-1].driving_command, dtype=torch.float32
        )
        status_feature = torch.cat(
            [
                high_command_one_hot.clone(),
                torch.tensor(ego_statuses[-1].ego_velocity, dtype=torch.float32),
                torch.tensor(ego_statuses[-1].ego_acceleration, dtype=torch.float32),
            ],
            dim=-1,
        )

        if not self.cache_hidden_state:
            image_path = str(cameras[-1].cam_f0.image)

            path_as_ordinals = [ord(char) for char in image_path]

            path_tensor = torch.tensor(path_as_ordinals, dtype=torch.long)

            return {
                "history_trajectory": history_trajectory.cpu(),
                "high_command_one_hot": high_command_one_hot.cpu(),
                "status_feature": status_feature.cpu(),
                "image_path_tensor": path_tensor.cpu(),
            }
        else:
            use_ori=False
            if use_ori:
                if self.backbone is None:
                    raise RuntimeError(
                        "FeatureBuilder is in online mode, but the backbone was not initialized."
                    )

                pixel_values = load_image(str(cameras[-1].cam_f0.image)).unsqueeze(0)

                pixel_values_squeezed = pixel_values.squeeze(1)
                num_patches_list = [pv.shape[0] for pv in pixel_values_squeezed]
                pixel_values_cat = torch.cat(list(pixel_values_squeezed), dim=0)

                navigation_commands = ["turn left", "go straight", "turn right"]
                command_str = next(
                    (
                        navigation_commands[i]
                        for i, v in enumerate(high_command_one_hot)
                        if v == 1
                    ),
                    "unknown",
                )
                history_str = " ".join(
                    [
                        f"   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})"
                        for i in range(4)
                    ]
                )

                prompt = f"<image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
                output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
                questions = [f"{prompt}{output_requirements}"]

                outputs = self.backbone(
                    pixel_values_cat.cuda(), questions, num_patches_list=num_patches_list
                )
                last_hidden_state = outputs.hidden_states[-1]

                return {
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                    "status_feature": status_feature.cpu(),
                }
            else:
                #our methods
                if self.backbone is None:
                    raise RuntimeError(
                        "FeatureBuilder is in online mode, but the backbone was not initialized."
                    )
                
                image_inputs = []
                for i in range(4):
                    image_inputs.append(cameras[i].cam_f0.image)
                pixel_values_list = []
                if type(image_inputs) == list:
                    for image_input in image_inputs:
                        pixel_values = load_image(image_input, max_num=4).to(torch.bfloat16)
                        pixel_values_list.append(pixel_values)
                pixel_values = torch.cat(pixel_values_list, dim=0)
                num_patches_list = [pv.size(0) for pv in pixel_values_list]

                navigation_commands = ["turn left", "go straight", "turn right"]
                command_str = "unknown"
                for i, v in enumerate(high_command_one_hot):
                    try:
                        if v == 1:
                            command_str = navigation_commands[i]
                            break
                    except IndexError:
                        print(f"[IndexError] navigation_commands out of bounds: i={i}, high_command_one_hot={high_command_one_hot.tolist()}")
                        break
                history_str = " ".join(
                    [
                        f"   - t-{3-i}: ({format_number(history_trajectory[i, 0].item())}, {format_number(history_trajectory[i, 1].item())}, {format_number(history_trajectory[i, 2].item())})"
                        for i in range(4)
                    ]
                )

                prompt = f"<image><image><image><image>\nAs an autonomous driving system, predict the vehicle's trajectory based on:\n1. Visual perception from front camera view (last 4 timesteps)\n2. Historical motion context (last 4 timesteps):{history_str}\n3. Active navigation command: [{command_str.upper()}]"
                output_requirements = "\nOutput requirements:\n- Predict 8 future trajectory points\n- Each point format: (x:float, y:float, heading:float)\n- Use [PT, ...] to encapsulate the trajectory\n- Maintain numerical precision to 2 decimal places"
                questions = [f"{prompt}{output_requirements}"]

                # outputs  = self.backbone(
                #     pixel_values.cuda(), questions, num_patches_list=num_patches_list
                # )
                # last_hidden_state = outputs['outputs'].hidden_states[-1]
                # occ_hidden_state = outputs['occ_out']
                # agent_hidden_state = outputs['agent_out']
                # gp_hidden_state = outputs['gp_out']

                # return {
                #     "history_trajectory": history_trajectory.cpu(),
                #     "high_command_one_hot": high_command_one_hot.cpu(),
                #     "last_hidden_state": last_hidden_state.squeeze(0).float().cpu(),
                #     "status_feature": status_feature.cpu(),
                #     "occ_hidden_state":occ_hidden_state.squeeze(0).float().cpu(),
                #     "agent_hidden_state":agent_hidden_state.squeeze(0).float().cpu(),
                #     "gp_hidden_state":gp_hidden_state.squeeze(0).float().cpu()
                # }

                outputs  = self.backbone(
                    pixel_values.cuda(), questions, num_patches_list=num_patches_list, ego_status=ego_statuses)
                def safe_get_dict(d, key, default=None):
                    return d[key] if key in d and d[key] is not None else default

                last_hidden_state  = safe_get_dict(outputs, 'outputs')
                if last_hidden_state is not None and hasattr(last_hidden_state, "hidden_states"):
                    last_hidden_state = last_hidden_state.hidden_states[-1]
                else:
                    last_hidden_state = None

                occ_hidden_state   = safe_get_dict(outputs, 'occ_out')
                agent_hidden_state = safe_get_dict(outputs, 'agent_out')
                gp_hidden_state    = safe_get_dict(outputs, 'gp_out')
                dream_occ_hidden_state   = safe_get_dict(outputs, 'dream_occ_out')
                dream_agent_hidden_state = safe_get_dict(outputs, 'dream_agent_out')

                return {
                    "history_trajectory": history_trajectory.cpu(),
                    "high_command_one_hot": high_command_one_hot.cpu(),
                    "last_hidden_state": None if last_hidden_state is None else last_hidden_state.squeeze(0).float().cpu(),
                    "status_feature": status_feature.cpu(),
                    "occ_hidden_state": None if occ_hidden_state is None else occ_hidden_state.squeeze(0).float().cpu(),
                    "agent_hidden_state": None if agent_hidden_state is None else agent_hidden_state.squeeze(0).float().cpu(),
                    "gp_hidden_state": None if gp_hidden_state is None else gp_hidden_state.squeeze(0).float().cpu(),
                    "dream_occ_hidden_state": None if dream_occ_hidden_state is None else dream_occ_hidden_state.squeeze(0).float().cpu(),
                    "dream_agent_hidden_state": None if dream_agent_hidden_state is None else dream_agent_hidden_state.squeeze(0).float().cpu(),
                }





class TrajectoryTargetBuilder(AbstractTargetBuilder):
    def __init__(self, trajectory_sampling: TrajectorySampling):
        self._trajectory_sampling = trajectory_sampling

    def get_unique_name(self) -> str:
        return "trajectory_target"

    def compute_targets(self, scene: Scene) -> Dict[str, torch.Tensor]:
        future_trajectory = scene.get_future_trajectory(
            num_trajectory_frames=self._trajectory_sampling.num_poses
        )
        return {"trajectory": torch.tensor(future_trajectory.poses)}
