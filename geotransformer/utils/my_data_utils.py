import numpy as np
import open3d as o3d
from typing import List, Optional
from numpy.typing import NDArray

def visualize_multiple_npy_point_clouds(npy_pointcloud_list:Optional[List[NDArray]], colors:Optional[List[List[float]]]=None) -> None:
    """
    複数の.npy点群を同時に可視化する。

    Args:
        npy_pointcloud_list (list of np.ndarray): 各点群の座標データ（N x 3のnumpy配列）。
        colors (list of list): 各点群の色（RGBリスト）。省略時は自動色。
    """
    pcds = []
    default_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 赤・緑・青
    for i, npy_points in enumerate(npy_pointcloud_list):
        points = npy_points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        pcd.paint_uniform_color(color)
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)
    
def getting_pointcloud_from_transform(pointcloud: NDArray, transform_matrix: NDArray) -> NDArray:
    """
    点群を変換行列で変換する。

    Args:
        pointcloud (np.ndarray): 点群の座標データ（N x 3のnumpy配列）。
        transform_matrix (np.ndarray): 変換行列（4 x 4のnumpy配列）。

    Returns:
        np.ndarray: 変換後の点群座標データ。
    """
    pointcloud_homogeneous = np.hstack((pointcloud, np.ones((pointcloud.shape[0], 1))))
    transformed_pointcloud = pointcloud_homogeneous @ transform_matrix.T # numpyは行ベクトルなので、右から転置を掛ける（p' = Tp, p'^T = (Tp)^T = p^(T)T^T）
    return transformed_pointcloud[:, :3]

# 変換行列の平行移動と角度誤差を計算する関数
def calculate_translation_and_angle_error_SE2(
    gt_transform: np.ndarray,
    estimated_transform: np.ndarray,
) -> NDArray:
    """
    Calculate translation and angle error between ground truth and estimated SE(2) transforms.

    Args:
        gt_transform (np.ndarray): Ground truth SE(2) transform matrix (4x4).
        estimated_transform (np.ndarray): Estimated SE(2) transform matrix (4x4).

    Returns:
        tuple: Translation error (float), angle error in degrees (float).
    """
    # Extract translation
    gt_translation = gt_transform[:2, 3]
    estimated_translation = estimated_transform[:2, 3]
    
    translation_error = np.linalg.norm(gt_translation - estimated_translation)
    translation_error = round(translation_error, 4)  # Round to 3 decimal places

    # Extract rotation angle in radians
    gt_angle = np.arctan2(gt_transform[1, 0], gt_transform[0, 0])
    estimated_angle = np.arctan2(estimated_transform[1, 0], estimated_transform[0, 0])
    
    angle_error = np.rad2deg(np.abs(gt_angle - estimated_angle))
    angle_error = round(angle_error, 4)  # Round to 3 decimal places
    
    total_error = np.array([translation_error, angle_error])

    return total_error

def calculate_transform_error_average(
    transform_errors: List[float, float],
) -> NDArray:
    """
    Calculate the average translation and angle error from a list of errors.

    Args:
        transform_errors (list of tuples): List of tuples containing translation and angle errors.

    Returns:
        tuple: Average translation error (float), average angle error in degrees (float).
    """
    if not transform_errors:
        return np.array([0.0, 0.0])

    avg_translation_error = np.mean([error[0] for error in transform_errors])
    avg_angle_error = np.mean([error[1] for error in transform_errors])

    return np.array([avg_translation_error, avg_angle_error])