import os
import sys
import json
import re

os.environ["NUPLAN_MAP_VERSION"] = "nuplan-maps-v1.0"
os.environ["NUPLAN_MAPS_ROOT"] = "/path/to/NAVSIM/dataset/maps"
os.environ["NAVSIM_EXP_ROOT"] = "/path/to/NAVSIM/exp"
os.environ["NAVSIM_DEVKIT_ROOT"] = "/path/to/NAVSIM/navsim-main"
os.environ["OPENSCENE_DATA_ROOT"] = "/path/to/NAVSIM/dataset"

os.environ["VLM_PATH"] = "/path/to/internvl_wm_model_save_dir"
os.environ["CACHE_PATH"] = "/path/to/sgdrive_agent_cache"
os.environ["METRIC_CACHE_PATH"] = "/path/to/metric_cache"

sys.path.append(os.environ["NAVSIM_DEVKIT_ROOT"])

from pathlib import Path

import hydra
from hydra.core.global_hydra import GlobalHydra
from hydra.utils import instantiate

if GlobalHydra.instance().is_initialized():
    GlobalHydra.instance().clear()

import matplotlib.pyplot as plt
import numpy as np
from nuplan.database.utils.pointclouds.lidar import LidarPointCloud
import io
import cv2

from navsim.agents.transfuser.transfuser_agent import TransfuserAgent
from navsim.agents.intervl_agent_wm import InternVLAgent_WM
from navsim.common.dataclasses import SceneFilter, SensorConfig
from navsim.common.dataloader import SceneLoader
from navsim.visualization.bev import add_annotations_to_bev_ax, add_lidar_to_bev_ax
from navsim.visualization.plots import (
    configure_bev_ax,
    frame_plot_to_gif,
    plot_bev_frame,
    plot_bev_with_agent,
    plot_cameras_frame,
    plot_cameras_frame_with_annotations,
    plot_cameras_frame_with_lidar,
)
from navsim.visualization.camera import visualize_gp_trajend_pred_gt_camera_bev

SPLIT = "mini"
FILTER = "navmini"
CONFIG_PATH_SCENE = (
    "../navsim/planning/script/config/common/train_test_split/scene_filter"
)
with hydra.initialize(config_path=CONFIG_PATH_SCENE, version_base=None):
    scene_filter_cfg = hydra.compose(config_name=FILTER)

CONFIG_PATH_AGENT = "../navsim/planning/script/config/common/agent"
CONFIG_NAME_AGENT = "internvl_agent_wm"
with hydra.initialize(config_path=CONFIG_PATH_AGENT, version_base=None):
    agent_cfg = hydra.compose(
        config_name=CONFIG_NAME_AGENT,
        overrides=["checkpoint_path=${oc.env:VLM_PATH}"],
    )

scene_filter: SceneFilter = instantiate(scene_filter_cfg)
openscene_data_root = Path(os.getenv("OPENSCENE_DATA_ROOT"))

scene_loader = SceneLoader(
    openscene_data_root / f"navsim_logs/{SPLIT}",
    openscene_data_root / f"sensor_blobs/{SPLIT}",
    scene_filter,
    sensor_config=SensorConfig.build_all_sensors(),
)

num_scenes_to_visualize = 10
tokens = np.random.choice(
    scene_loader.tokens, size=num_scenes_to_visualize, replace=False
)

tokens = scene_loader.tokens[: num_scenes_to_visualize]

exp_root = Path(os.environ["NAVSIM_EXP_ROOT"])

point_cloud_range = [0, -25.0, -4.0, 50.0, 25.0, 4.0]
def create_mask_with_opencv(points, distance=2, resolution=0.1):
    upper_points = [(x, y + distance) for x, y in points]
    lower_points = [(x, y - distance) for x, y in points][::-1]
    polygon_points = upper_points + lower_points

    width = 100
    height = 100

    mask = np.ones((height, width), dtype=np.uint8)
    
    polygon_pixels = np.array([[
                                [int((y - (point_cloud_range[1])) / resolution), int((x - point_cloud_range[0]) / resolution)] 
                                for x, y in polygon_points
                            ]], dtype=np.int32)
    
    polygon_pixels = np.array([polygon_pixels], dtype=np.int32)
    
    cv2.fillPoly(mask, polygon_pixels, 0)
    
    return mask

def get_occ(data_item):
    lidar_path = data_item['lidar_path'][2]
    new_prefix = "/path/to/NAVSIM/dataset/"
    lidar_path = re.sub(r'^.*?(?=sensor_blobs)', new_prefix, lidar_path)
    
    traj = np.array(data_item['future_trajectory'])
    with open(lidar_path, "rb") as fp:
        lidar_bytes = io.BytesIO(fp.read())
    lidar_points = LidarPointCloud.from_buffer(lidar_bytes, "pcd").points
    pts = np.column_stack((lidar_points[0],lidar_points[1], lidar_points[2], lidar_points[-1]))
    valid = (
        (pts[:, 0] >= point_cloud_range[0])
        & (pts[:, 0] < point_cloud_range[3])
        & (pts[:, 1] >= point_cloud_range[1])
        & (pts[:, 1] < point_cloud_range[4])
        & (pts[:, 2] >= point_cloud_range[2])
        & (pts[:, 2] < point_cloud_range[5])
    )
    valid_pts = pts[valid]
    voxel_coords = (
        (
            valid_pts[:, :3]
            - np.array(
                [
                    point_cloud_range[0],
                    point_cloud_range[1],
                    point_cloud_range[2],
                ]
            )
        )
        / (0.5)
    ).astype(np.int32)
    
    voxel_coords, inv_ind, voxel_counts = np.unique(
        voxel_coords, axis=0, return_inverse=True, return_counts=True
    )

    target_mask = np.zeros([100, 100, 16])
    target_mask[voxel_coords[:, 0], voxel_coords[:, 1], voxel_coords[:, 2]] = 1
    target = target_mask.astype(np.float32)
    traj = np.vstack([np.array([0,0,0]), traj])
    traj_mask = create_mask_with_opencv(traj[...,:2], distance=10, resolution=0.5)
    traj_mask = np.repeat(traj_mask[:, :, np.newaxis], 16, axis=2)
    target[traj_mask==1] = 0

    return target

scene2anno = dict()
jsonl_path = "/path/to/navsim_traj_world_front_view.jsonl"
with open(jsonl_path, "r") as f:
    raw_data = f.readlines()
    for line in raw_data:
        data_dict = json.loads(line)
        scene2anno[data_dict["token"]] = data_dict

for token in tokens:
    if token not in scene2anno:
        continue
    future_trajectory = np.array(scene2anno[token]["future_trajectory"])
    future_trajectory_xy = future_trajectory[:,:2]
    gp_gt = future_trajectory_xy[-1]

    scene = scene_loader.get_scene_from_token(token)

    output_dir = exp_root / f"visual"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_indices = [idx for idx in range(len(scene.frames))]  # all frames in scene

    # --- BEV Frame GIF ---
    # file_name_bev = output_dir / f"{token}_bev_frame.gif"
    # frame_plot_to_gif(file_name_bev, plot_bev_frame, scene, frame_indices)

    # --- BEV with Agent PNG ---
    # Note: plot_bev_with_agent does not vary by frame, so we plot the current state.
    agent = instantiate(agent_cfg)
    agent.initialize()
    # agent_input = scene_loader.get_agent_input_from_token(token)
    # breakpoint()
    agent_input = scene.get_agent_input()
    res = agent.compute_res(agent_input, output_text=True)

    if "boxes" in res:
        anns = res["boxes_dream"]
        # --- Cameras with Annotations GIF ---
        demo_cameras_gt = output_dir / f"{token}_cameras_gt.jpg"
        demo_cameras = output_dir / f"{token}_cameras_50queries.jpg"
        demo_cameras_comparison = output_dir / f"{token}_cameras_demo.jpg"

        # 3 history frames, the 4th is the current frame.
        frame = scene.frames[5]

        # import matplotlib.pyplot as plt
        # from navsim.visualization.camera import add_annotations_to_camera_ax

        # # Create a Figure, two subplots (1 row 2 columns)
        # fig, axes = plt.subplots(1, 2, figsize=(16, 6))  # Double the width

        # # First subplot: GT
        # ax = axes[0]
        # add_annotations_to_camera_ax(ax, frame.cameras.cam_f0, frame.annotations)
        # ax.axis("off")
        # ax.set_title("GT")  # Title

        # # Second subplot: Prediction
        # ax = axes[1]
        # add_annotations_to_camera_ax(ax, frame.cameras.cam_f0, anns)
        # ax.axis("off")
        # ax.set_title("Prediction")

        # # Compact layout
        # fig.tight_layout()

        # # Save to file
        # plt.savefig(demo_cameras_comparison, bbox_inches='tight', pad_inches=0)
        # plt.close(fig)
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
        from navsim.visualization.camera import add_annotations_to_camera_ax
        import textwrap

        # Create a Figure, two subplots (1 row 2 columns)
        fig = plt.figure(figsize=(24, 6), constrained_layout=True) # Triple the width

        gs = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 2])

        # --- 3. First subplot (GT) ---
        ax0 = fig.add_subplot(gs[0, 0])
        add_annotations_to_camera_ax(ax0, frame.cameras.cam_f0, frame.annotations)
        ax0.axis("off")
        ax0.set_title("GT", fontsize=16)  # Unified font size

        # Second subplot: Prediction
        ax1 = fig.add_subplot(gs[0, 1])
        add_annotations_to_camera_ax(ax1, frame.cameras.cam_f0, anns)
        ax1.axis("off")
        ax1.set_title("Prediction", fontsize=16)  # Unified font size
        
        if "text" in res:
            text = res["text"]
        # Third subplot (for text)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title("Description", fontsize=16)
        ax2.axis("off")  # Hide borders
        if text is not None:
            # (Key) Set y coordinate to 0.95 (starting *below title*)
            ax2.text(
                0.0, 0.95,                  # Text starts from top-left (0.0, 0.95) of ax2
                text,
                transform=ax2.transAxes,    # Coordinate system relative to ax2
                fontsize=12,                # You can adjust this
                color="black",
                va="top", ha="left",         # Vertical top alignment
                wrap=True                   # Auto wrap
            )
        ax2.axis("off")  # Hide text subplot borders
        plt.savefig(demo_cameras_comparison, bbox_inches='tight', pad_inches=0)
        plt.close(fig)

    if "occ" in res:
        save_path = output_dir / f"{token}_occ.npz"
        gt = get_occ(scene2anno[token])
        result = {
            "gt": gt,
            "pred": res["occ_dream"]
        }

        np.savez_compressed(save_path, data=result)

    if "gp" in res:
        gp_result = res["gp"].flatten()
        gp_result_3d = np.append(gp_result, 0)
        gp_gt_3d = np.append(gp_gt, 0)
        frame = scene.frames[3].cameras.cam_f0

        visualize_gp_trajend_pred_gt_camera_bev(
            camera=frame,
            pred_point_lidar=gp_result_3d,
            gt_point_lidar=gp_gt_3d,
            save_path=output_dir / f"{token}_gp.jpg"
        )