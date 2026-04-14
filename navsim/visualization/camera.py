from typing import List, Optional, Tuple,Dict, Any

import cv2
import numpy as np
import numpy.typing as npt
from PIL import ImageColor
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
import matplotlib.patches as patches
import torch
from navsim.common.dataclasses import Camera, Lidar, Annotations, Trajectory
from navsim.common.enums import LidarIndex, BoundingBoxIndex
from navsim.visualization.config import AGENT_CONFIG
from navsim.visualization.lidar import filter_lidar_pc, get_lidar_pc_color
from navsim.planning.scenario_builder.navsim_scenario_utils import tracked_object_types
import os
import matplotlib
from matplotlib.patches import Rectangle

def add_camera_ax(ax: plt.Axes, camera: Camera) -> plt.Axes:
    """
    Adds camera image to matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :return: ax object with image
    """
    ax.imshow(camera.image)
    return ax


def add_lidar_to_camera_ax(ax: plt.Axes, camera: Camera, lidar: Lidar) -> plt.Axes:
    """
    Adds camera image with lidar point cloud on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image
    """

    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print(lidar_pc.shape)
    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )
    print(image_height, image_width)
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    print(pc_in_cam.shape)
    for (x, y), color in zip(pc_in_cam, lidar_pc_colors[pc_in_fov_mask]):
        color = (int(color[0]), int(color[1]), int(color[2]))
        cv2.circle(image, (int(x), int(y)), 5, color, -1)

    ax.imshow(image)
    return ax

def dense_map(Pts, n, m, grid):
    """
    Generate a dense depth map from the sparse points.
    :param Pts: The sparse depth points (x, y, depth)
    :param n: Image width
    :param m: Image height
    :param grid: Neighborhood grid size for interpolation
    :return: Dense depth map
    """
    ng = 2 * grid + 1
    mX = np.zeros((m, n)) + np.float("inf")
    mY = np.zeros((m, n)) + np.float("inf")
    mD = np.zeros((m, n))
    
    # Fill the sparse depth points into the mX, mY, and mD matrices
    mX[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[0] - np.round(Pts[0])
    mY[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[1] - np.round(Pts[1])
    mD[np.int32(Pts[1]), np.int32(Pts[0])] = Pts[2]
    
    KmX = np.zeros((ng, ng, m - ng, n - ng))
    KmY = np.zeros((ng, ng, m - ng, n - ng))
    KmD = np.zeros((ng, ng, m - ng, n - ng))
    
    for i in range(ng):
        for j in range(ng):
            KmX[i, j] = mX[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmY[i, j] = mY[i: (m - ng + i), j: (n - ng + j)] - grid - 1 + i
            KmD[i, j] = mD[i: (m - ng + i), j: (n - ng + j)]
    
    S = np.zeros_like(KmD[0, 0])
    Y = np.zeros_like(KmD[0, 0])
    
    for i in range(ng):
        for j in range(ng):
            s = 1 / np.sqrt(KmX[i, j] * KmX[i, j] + KmY[i, j] * KmY[i, j])
            Y = Y + s * KmD[i, j]
            S = S + s
    
    S[S == 0] = 1
    out = np.zeros((m, n))
    out[grid + 1: -grid, grid + 1: -grid] = Y / S
    return out

def add_lidar_to_camera_ax_with_depth(
    ax: plt.Axes, camera: Camera, lidar: Lidar
) -> Tuple[plt.Axes, npt.NDArray[np.float32]]:
    """
    Adds camera image with lidar point cloud on matplotlib ax object and generates depth map.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param lidar: navsim lidar dataclass
    :return: ax object with image and the dense depth map
    """
    
    # Get image and LIDAR point cloud
    image, lidar_pc = camera.image.copy(), lidar.lidar_pc.copy()
    image_height, image_width = image.shape[:2]
    print("LIDAR point cloud shape:", lidar_pc.shape)

    lidar_pc = filter_lidar_pc(lidar_pc)
    lidar_pc_colors = np.array(get_lidar_pc_color(lidar_pc))

    pc_in_cam, pc_in_fov_mask, depth_values = _transform_pcs_to_images_with_depth(
        lidar_pc,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=(image_height, image_width),
    )

    print("Image dimensions:", image_height, image_width)
    print("Number of points in field of view:", np.sum(pc_in_fov_mask))

    # Keep only points in field of view
    pc_in_cam = pc_in_cam[pc_in_fov_mask]
    depth_values = depth_values[pc_in_fov_mask]
    lidar_pc_colors = lidar_pc_colors[pc_in_fov_mask]

    # Create empty depth map initialized to infinity
    depth_map_intermediate = np.full((image_height, image_width), np.inf)

    # Project LIDAR points to image plane and draw
    for (x, y), depth, color in zip(pc_in_cam[:, 0:2], depth_values, lidar_pc_colors):
        color = (int(color[0]), int(color[1]), int(color[2]))
        x, y = int(x), int(y)
        if 0 <= x < image_width and 0 <= y < image_height:
            color = (int(color[0]), int(color[1]), int(color[2]))
            cv2.circle(image, (int(x), int(y)), 5, color, -1)
            depth_map_intermediate[y, x] = min(depth_map_intermediate[y, x], depth)

    ax.imshow(image)

    # Print depth map min/max values to check range
    valid_depth_values = depth_map_intermediate[depth_map_intermediate != np.inf]

    print("Depth map min value:", np.min(valid_depth_values))
    print("Depth map max value:", np.max(valid_depth_values))

    # Replace np.inf with max valid depth in depth map
    if len(valid_depth_values) > 0:
        max_depth = np.max(valid_depth_values)
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth
    else:
        max_depth = 0  # Use 0 as max depth if no valid depth values exist
        depth_map_intermediate[np.isinf(depth_map_intermediate)] = max_depth

    # Generate dense depth map using dense_map function
    dense_depth_map = dense_map(
        np.array([pc_in_cam[:, 0], pc_in_cam[:, 1], depth_values]),
        image_width,
        image_height,
        grid=8  # Adjust this value to control smoothness
    )

    # Normalize depth map to 0-255
    dense_depth_map_normalized = cv2.normalize(dense_depth_map, None, 0, 255, cv2.NORM_MINMAX)

    # Apply 'Spectral_r' colormap to dense depth map
    colormap = plt.get_cmap('Spectral_r')
    colored_dense_depth_map = colormap(dense_depth_map_normalized / 255.0)

    # Convert RGBA to RGB
    colored_dense_depth_map = (colored_dense_depth_map[:, :, :3] * 255).astype(np.uint8)

    # Get filename and output directory
    outdir = './output_dense_depth_maps'
    os.makedirs(outdir, exist_ok=True)
    filename = "lidar_dense_depth_image"

    # Save dense depth map
    cv2.imwrite(os.path.join(outdir, filename + '_dense_depth.png'), colored_dense_depth_map)

    # Combine image and dense depth map
    split_region = np.ones((image.shape[0], 50, 3), dtype=np.uint8) * 255
    combined_result = cv2.hconcat([image, split_region, colored_dense_depth_map])

    # Save combined result
    cv2.imwrite(os.path.join(outdir, filename + '_combined.png'), combined_result)

    return ax, dense_depth_map

def add_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with bounding boxes on matplotlib ax object
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])
    
    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Then draw project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]
    image = _plot_rect_3d_on_img(camera.image.copy(), box_corners, box_labels)

    ax.imshow(image)
    return ax

def add_2d_annotations_to_camera_ax(ax: plt.Axes, camera: Camera, annotations: Annotations) -> plt.Axes:
    """
    Adds camera image with 2D bounding boxes on matplotlib ax object.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param annotations: navsim annotations dataclass
    :return: ax object with image
    """

    box_labels = annotations.names
    boxes = _transform_annotations_to_camera(
        annotations.boxes,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
    )
    box_positions, box_dimensions, box_heading = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.DIMENSION],
        boxes[:, BoundingBoxIndex.HEADING],
    )
    corners_norm = np.stack(np.unravel_index(np.arange(8), [2] * 3), axis=1)
    corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array([0.5, 0.5, 0.5])
    corners = box_dimensions.reshape([-1, 1, 3]) * corners_norm.reshape([1, 8, 3])

    corners = _rotation_3d_in_axis(corners, box_heading, axis=1)
    corners += box_positions.reshape(-1, 1, 3)

    # Project corners to image.
    box_corners, corners_pc_in_fov = _transform_points_to_image(corners.reshape(-1, 3), camera.intrinsics)
    box_corners = box_corners.reshape(-1, 8, 2)
    corners_pc_in_fov = corners_pc_in_fov.reshape(-1, 8)
    valid_corners = corners_pc_in_fov.any(-1)

    box_corners, box_labels = box_corners[valid_corners], box_labels[valid_corners]

    # Calculate 2D bounding boxes from projected 3D corners.
    box_2d_list = []
    for corner_set in box_corners:
        min_x = np.min(corner_set[:, 0])
        max_x = np.max(corner_set[:, 0])
        min_y = np.min(corner_set[:, 1])
        max_y = np.max(corner_set[:, 1])
        box_2d_list.append([min_x, min_y, max_x, max_y])

    box_2d_list = np.array(box_2d_list)
    print(box_2d_list)
    image = _plot_rect_2d_on_img(camera.image.copy(), box_2d_list, box_labels)

    ax.imshow(image)
    return ax

def _plot_rect_2d_on_img(image: np.ndarray, boxes_2d: np.ndarray, labels: list, color=(0, 255, 0), thickness=2) -> np.ndarray:
    """
    Draws 2D bounding boxes on an image.
    :param image: input image (numpy array)
    :param boxes_2d: 2D bounding boxes (numpy array), each box is [x_min, y_min, x_max, y_max]
    :param labels: list of labels for each bounding box
    :param color: color of the bounding box (BGR format)
    :param thickness: thickness of the bounding box lines
    :return: image with bounding boxes drawn
    """

    image_copy = image.copy()

    if boxes_2d is not None and len(boxes_2d) > 0:
        # if labels is None:
        #   labels = []
        for i, box in enumerate(boxes_2d):
            x_min, y_min, x_max, y_max = map(int, box)  # Convert coordinates to integers
            cv2.rectangle(image_copy, (x_min, y_min), (x_max, y_max), color, thickness)

            # # Add label
            # if labels and i < len(labels):
            #     label = labels[i]
            #     cv2.putText(image_copy, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)

    return image_copy


def _transform_annotations_to_camera(
    boxes: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """
    Helper function to transform bounding boxes into camera frame
    TODO: Refactor
    :param boxes: array representation of bounding boxes
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :return: bounding boxes in camera coordinates
    """

    locs, rots = (
        boxes[:, BoundingBoxIndex.POSITION],
        boxes[:, BoundingBoxIndex.HEADING :],
    )
    dims_cam = boxes[
        :, [BoundingBoxIndex.LENGTH, BoundingBoxIndex.HEIGHT, BoundingBoxIndex.WIDTH]
    ]  # l, w, h -> l, h, w

    rots_cam = np.zeros_like(rots)
    for idx, rot in enumerate(rots):
        rot = Quaternion(axis=[0, 0, 1], radians=rot)
        rot = Quaternion(matrix=sensor2lidar_rotation).inverse * rot
        rots_cam[idx] = -rot.yaw_pitch_roll[0]

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    locs_cam = np.concatenate([locs, np.ones_like(locs)[:, :1]], -1)  # -1, 4
    locs_cam = lidar2cam_rt.T @ locs_cam.T
    locs_cam = locs_cam.T
    locs_cam = locs_cam[:, :-1]
    return np.concatenate([locs_cam, dims_cam, rots_cam], -1)


def _rotation_3d_in_axis(points: npt.NDArray[np.float32], angles: npt.NDArray[np.float32], axis: int = 0):
    """
    Rotate 3D points by angles according to axis.
    TODO: Refactor
    :param points: array of points
    :param angles: array of angles
    :param axis: axis to perform rotation, defaults to 0
    :raises value: _description_
    :raises ValueError: if axis invalid
    :return: rotated points
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, zeros, -rot_sin]),
                np.stack([zeros, ones, zeros]),
                np.stack([rot_sin, zeros, rot_cos]),
            ]
        )
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack(
            [
                np.stack([rot_cos, -rot_sin, zeros]),
                np.stack([rot_sin, rot_cos, zeros]),
                np.stack([zeros, zeros, ones]),
            ]
        )
    elif axis == 0:
        rot_mat_T = np.stack(
            [
                np.stack([zeros, rot_cos, -rot_sin]),
                np.stack([zeros, rot_sin, rot_cos]),
                np.stack([ones, zeros, zeros]),
            ]
        )
    else:
        raise ValueError(f"axis should in range [0, 1, 2], got {axis}")
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def _plot_rect_3d_on_img(
    image: npt.NDArray[np.float32],
    box_corners: npt.NDArray[np.float32],
    box_labels: List[str],
    thickness: int = 3,
) -> npt.NDArray[np.uint8]:
    """
    Plot the boundary lines of 3D rectangular on 2D images.
    TODO: refactor
    :param image:  The numpy array of image.
    :param box_corners: Coordinates of the corners of 3D, shape of [N, 8, 2].
    :param box_labels: labels of boxes for coloring
    :param thickness: pixel width of liens, defaults to 3
    :return: image with 3D bounding boxes
    """
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )
    for i in range(len(box_corners)):
        layer = tracked_object_types[box_labels[i]]
        color = ImageColor.getcolor(AGENT_CONFIG[layer]["fill_color"], "RGB")
        corners = box_corners[i].astype(int)
        for start, end in line_indices:
            cv2.line(
                image,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    return image.astype(np.uint8)


def _transform_points_to_image(
    points: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    image_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param points: points in camera frame
    :param intrinsic: camera intrinsics
    :param image_shape: shape of image in pixel
    :param eps: lower threshold of points, defaults to 1e-3
    :return: points in pixel coordinates, mask of values in frame
    """
    points = points[:, :3]

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic

    pc_img = np.concatenate([points, np.ones_like(points)[:, :1]], -1)
    pc_img = viewpad @ pc_img.T
    pc_img = pc_img.T

    cur_pc_in_fov = pc_img[:, 2] > eps
    pc_img = pc_img[..., 0:2] / np.maximum(pc_img[..., 2:3], np.ones_like(pc_img[..., 2:3]) * eps)
    if image_shape is not None:
        img_h, img_w = image_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (pc_img[:, 0] < (img_w - 1))
            & (pc_img[:, 0] > 0)
            & (pc_img[:, 1] < (img_h - 1))
            & (pc_img[:, 1] > 0)
        )
    return pc_img, cur_pc_in_fov


def _transform_pcs_to_images(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_]]:
    """
    Transforms points in camera frame to image pixel coordinates
    TODO: refactor
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame
    """
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], -1)
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps
    cur_pc_cam = cur_pc_cam[..., 0:2] / np.maximum(cur_pc_cam[..., 2:3], np.ones_like(cur_pc_cam[..., 2:3]) * eps)

    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )
    return cur_pc_cam, cur_pc_in_fov


def _transform_pcs_to_images_with_depth(
    lidar_pc: npt.NDArray[np.float32],
    sensor2lidar_rotation: npt.NDArray[np.float32],
    sensor2lidar_translation: npt.NDArray[np.float32],
    intrinsic: npt.NDArray[np.float32],
    img_shape: Optional[Tuple[int, int]] = None,
    eps: float = 1e-3,
) -> Tuple[npt.NDArray[np.float32], npt.NDArray[np.bool_], npt.NDArray[np.float32]]:
    """
    Transforms points in camera frame to image pixel coordinates and returns depth values.
    :param lidar_pc: lidar point cloud
    :param sensor2lidar_rotation: camera rotation
    :param sensor2lidar_translation: camera translation
    :param intrinsic: camera intrinsics
    :param img_shape: image shape in pixels, defaults to None
    :param eps: threshold for lidar pc height, defaults to 1e-3
    :return: lidar pc in pixel coordinates, mask of values in frame, depth values in frame
    """
    # Get (x, y, z) coordinates of LIDAR point cloud
    pc_xyz = lidar_pc[LidarIndex.POSITION, :].T
    print(f"pc_xyz shape: {pc_xyz.shape}")  # Debug: Check shape of lidar point cloud

    lidar2cam_r = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t = sensor2lidar_translation @ lidar2cam_r.T
    lidar2cam_rt = np.eye(4)
    lidar2cam_rt[:3, :3] = lidar2cam_r.T
    lidar2cam_rt[3, :3] = -lidar2cam_t

    viewpad = np.eye(4)
    viewpad[: intrinsic.shape[0], : intrinsic.shape[1]] = intrinsic
    lidar2img_rt = viewpad @ lidar2cam_rt.T

    # Add homogeneous coordinates to make 4D vector (x, y, z, 1)
    cur_pc_xyz = np.concatenate([pc_xyz, np.ones_like(pc_xyz)[:, :1]], axis=-1)
    print(f"cur_pc_xyz shape after concat: {cur_pc_xyz.shape}")  # Debug: Check shape after concat

    # Project to camera coordinate system
    cur_pc_cam = lidar2img_rt @ cur_pc_xyz.T
    cur_pc_cam = cur_pc_cam.T
    print(f"cur_pc_cam shape after projection: {cur_pc_cam.shape}")  # Debug: Check shape after projection

    # Filter out points outside FOV
    cur_pc_in_fov = cur_pc_cam[:, 2] > eps

    # Standardize homogeneous coordinates: divide by Z to get 2D coordinates
    cur_pc_cam[:, 0:2] = cur_pc_cam[:, 0:2] / np.maximum(cur_pc_cam[:, 2:3], np.ones_like(cur_pc_cam[:, 2:3]) * eps)
    print(f"cur_pc_cam shape after normalization: {cur_pc_cam.shape}")  # Debug: Check shape after normalization

    # Get depth values (Z coordinates)
    depth_values = cur_pc_cam[:, 2]
    print(f"depth_values shape: {depth_values.shape}")  # Debug: Check depth_values shape

    # Keep only points within FOV
    if img_shape is not None:
        img_h, img_w = img_shape
        cur_pc_in_fov = (
            cur_pc_in_fov
            & (cur_pc_cam[:, 0] < (img_w - 1))
            & (cur_pc_cam[:, 0] > 0)
            & (cur_pc_cam[:, 1] < (img_h - 1))
            & (cur_pc_cam[:, 1] > 0)
        )

    return cur_pc_cam, cur_pc_in_fov, depth_values


def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
    """
    Add trajectory poses as connected lines on camera image, with an arrow at the front.
    If the first point is outside the image, calculate intersection with boundary as start.
    :param ax: matplotlib ax object
    :param camera: navsim camera dataclass
    :param trajectory: navsim trajectory dataclass
    :param config: dictionary with plotting parameters
    :return: ax with plot
    """
    poses_2d = trajectory.poses[:, :2]
    poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1))], axis=1)
    poses = np.concatenate([np.array([[0, 0, 0]]), poses_3d])

    pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
        poses.T,
        camera.sensor2lidar_rotation,
        camera.sensor2lidar_translation,
        camera.intrinsics,
        img_shape=camera.image.shape[:2]
    )

    image_height, image_width = camera.image.shape[:2]
    valid_cam_points = pc_in_cam[pc_in_fov_mask]

    points_to_plot = []
    first_point = pc_in_cam[1] if len(pc_in_cam) > 1 else None
    second_point = pc_in_cam[2] if len(pc_in_cam) > 2 else None

    # Check if first point is inside image
    if first_point is not None and pc_in_fov_mask[1]:
        points_to_plot.append(first_point)
    elif first_point is not None and second_point is not None:
        # If outside and second point exists, calculate intersection
        start_pt = first_point
        end_pt = second_point
        intersection_point = _get_intersection_with_image_bottom_boundary(start_pt, end_pt, image_width, image_height)
        if intersection_point is not None:
            points_to_plot.append(intersection_point)

    # Add remaining trajectory points
    for i in range(2, len(pc_in_cam)):
        if pc_in_fov_mask[i]:
            points_to_plot.append(pc_in_cam[i])

    if len(points_to_plot) > 1:
        plot_points = np.array(points_to_plot)
        # Connect valid camera points into lines
        ax.plot(
            plot_points[:, 0],
            plot_points[:, 1],
            color=config["line_color"],
            alpha=config["line_color_alpha"],
            linewidth=config["line_width"],
            linestyle=config["line_style"],
            marker=config.get("marker", None),
            markersize=config.get("marker_size", 0),
            markeredgecolor=config.get("marker_edge_color", None),
            zorder=config["zorder"],
        )

        # Add arrow pointing to last segment
        last_point = plot_points[-1]
        second_last_point = plot_points[-2]

        arrow_dx = last_point[0] - second_last_point[0]
        arrow_dy = last_point[1] - second_last_point[1]

        # Calculate arrow end point by extending along trajectory direction
        arrow_end_x = last_point[0] + arrow_dx
        arrow_end_y = last_point[1] + arrow_dy

        arrow = patches.FancyArrowPatch(
            posA=(last_point[0], last_point[1]),
            posB=(arrow_end_x, arrow_end_y),
            arrowstyle='-|>',
            mutation_scale=15,
            fc=config["arrow_color"],
            ec=config["arrow_edge_color"],
            alpha=config["arrow_alpha"],
            linewidth=config.get("arrow_line_width", config["line_width"]),
            connectionstyle='arc3,rad=0.0',
            zorder=config["zorder"] + 1,
        )
        ax.add_patch(arrow)

    return ax


# from scipy.interpolate import splprep, splev

# def add_trajectory_to_camera_ax(ax: plt.Axes, camera: Camera, trajectory: Trajectory, config: Dict[str, Any]) -> plt.Axes:
#     poses_2d = trajectory.poses[:, :2]
#     poses_3d = np.concatenate([poses_2d, np.zeros((poses_2d.shape[0], 1))], axis=1)
#     poses = np.concatenate([np.array([[0, 0, 0]]), poses_3d])

#     pc_in_cam, pc_in_fov_mask = _transform_pcs_to_images(
#         poses.T,
#         camera.sensor2lidar_rotation,
#         camera.sensor2lidar_translation,
#         camera.intrinsics,
#         img_shape=camera.image.shape[:2]
#     )

#     image_height, image_width = camera.image.shape[:2]

#     # === Start point handling: ensure trajectory enters from FOV or boundary ===
#     points_to_plot = []
#     first_point = pc_in_cam[1] if len(pc_in_cam) > 1 else None
#     second_point = pc_in_cam[2] if len(pc_in_cam) > 2 else None

#     if first_point is not None and pc_in_fov_mask[1]:
#         points_to_plot.append(first_point)
#     elif first_point is not None and second_point is not None:
#         intersection_point = _get_intersection_with_image_bottom_boundary(
#             first_point, second_point, image_width, image_height
#         )
#         if intersection_point is not None:
#             points_to_plot.append(intersection_point)

#     for i in range(2, len(pc_in_cam)):
#         if pc_in_fov_mask[i]:
#             points_to_plot.append(pc_in_cam[i])

#     valid_points = np.array(points_to_plot)

#     if len(valid_points) < 2:
#         return ax

#     # === Spline smooth trajectory ===
#     x, y = valid_points[:, 0], valid_points[:, 1]
#     try:
#         tck, u = splprep([x, y], s=2)
#         u_fine = np.linspace(0, 1, 200)
#         x_smooth, y_smooth = splev(u_fine, tck)
#     except Exception:
#         # Fallback to direct line if not enough points
#         x_smooth, y_smooth = x, y

#     # === Clean lines (no glow, no arrow) ===
#     ax.plot(x_smooth, y_smooth,
#             color=config["line_color"],
#             alpha=config["line_color_alpha"],
#             linewidth=config["line_width"],
#             linestyle=config["line_style"],
#             zorder=config["zorder"])

#     return ax


def _get_intersection_with_image_bottom_boundary(origin_point, end_point, image_width, image_height):
    """
    Calculate intersection of line segment with image *bottom* boundary.
    :param origin_point: Line start point (numpy array [x, y])
    :param end_point: Line end point (numpy array [x, y])
    :param image_width: Image width
    :param image_height: Image height
    :return: Intersection with bottom boundary (numpy array [x, y]), or None if no valid intersection
    """
    x1, y1 = origin_point
    x2, y2 = end_point

    # Image bottom boundary y = y_max
    x_min, x_max = 0, image_width - 1
    y_max = image_height - 1


    # 1. Intersection with bottom boundary y = y_max
    if y1 != y2: # Avoid division by zero
        x_intersection_bottom = x1 + (y_max - y1) * (x2 - x1) / (y2 - y1)
        if x_min <= x_intersection_bottom <= x_max and min(y1, y2) <= y_max <= max(y1, y2):
            return np.array([x_intersection_bottom, y_max])

    return None


def lidar_to_carema_point(lidar_point, sensor2lidar_rotation, sensor2lidar_translation):
    lidar_point=np.array(lidar_point)
    point_h=np.concatenate([lidar_point,[1.0]])
    lidar2cam_r=np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_t=sensor2lidar_translation@lidar2cam_r.T
    lidar2cam_rt=np.eye(4)
    lidar2cam_rt[:3,:3]=lidar2cam_r.T
    lidar2cam_rt[3,:3]=-lidar2cam_t
    point_cam=(lidar2cam_rt.T@point_h.T).T
    return point_cam[:3]


def _plot_points_on_camera_ax(ax, camera, points_info):
    ax.imshow(camera.image)
    ax.axis("off")
    
    for lidar_point, color, label, marker, is_hollow in points_info:
        cam_point = lidar_to_carema_point(lidar_point, camera.sensor2lidar_rotation, camera.sensor2lidar_translation)
        pixel, mask = _transform_points_to_image(np.array([cam_point]), camera.intrinsics, image_shape=camera.image.shape[:2])
        
        if mask[0]:
            if is_hollow:
                 ax.scatter(pixel[0,0], pixel[0,1], s=60, edgecolors=color, facecolors="none", linewidths=2, label=label)
            else:
                 ax.scatter(pixel[0,0], pixel[0,1], s=80, marker=marker, c=color, label=label)


def visualize_gp_trajend_pred_gt_camera_bev(camera, pred_point_lidar, gt_point_lidar, ego_length=4.5, ego_width=2.0, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    points_info = [
        (pred_point_lidar, "red", "prediction by gp head", None, True),                   
        (gt_point_lidar, "lime", "GT", "*", False)
    ]

    # Camera View
    ax_cam = axes[0]
    ax_cam.set_title("Camera View")
    _plot_points_on_camera_ax(ax_cam, camera, points_info)
    
    ax_bev = axes[1]
    ax_bev.set_title("BEV view\n(X=forward, Y=left)")
    ax_bev.set_xlabel("Y (m, left positive)")
    ax_bev.set_ylabel("X (m, forward positive)")
    ax_bev.set_xlim([-32,32])
    ax_bev.invert_xaxis()
    ax_bev.set_ylim([-8,64])
    ax_bev.grid(True)
    ax_bev.set_aspect("equal")
    
    car_rect=Rectangle((-ego_width/2,ego_width/2),ego_width,ego_length,edgecolor="black",facecolor="gray",alpha=0.5)
    ax_bev.add_patch(car_rect)
    
    ax_bev.scatter(pred_point_lidar[1],pred_point_lidar[0],s=60,edgecolors="red",facecolors="none",linewidths=2,label="prediction by gp head")
    ax_bev.scatter(gt_point_lidar[1],gt_point_lidar[0],s=80,marker="*",c="lime",label="GT")
    ax_bev.legend()
    
    pred_text=(f"Prediction Point by gp head (X forward, Y left, Z up): "
               f"({pred_point_lidar[0]:.2f} m, {pred_point_lidar[1]:.2f} m, {pred_point_lidar[2]:.2f} m)")
    gt_text=(f"GT Point (X forward, Y left, Z up): "
             f"({gt_point_lidar[0]:.2f} m, {gt_point_lidar[1]:.2f} m, {gt_point_lidar[2]:.2f} m)")
    
    fig.text(0.5, 0.05, pred_text, ha="center", fontsize=10, color="red")
    fig.text(0.5, 0.02, gt_text, ha="center", fontsize=10, color="lime")
    
    plt.tight_layout(rect=[0, 0.075, 1, 1])
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.close(fig)
    