import numpy as np
import open3d as o3d
from typing import List, Optional
from numpy.typing import NDArray

def visualize_multiple_npy_point_clouds(npy_paths:Optional[List[str]], colors:Optional[List[List[float]]]=None) -> None:
    """
    複数の.npy点群を同時に可視化する。

    Args:
        npy_paths (list of str): 点群ファイルパスのリスト
        colors (list of list): 各点群の色（RGBリスト）。省略時は自動色。
    """
    pcds = []
    default_colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # 赤・緑・青
    for i, path in enumerate(npy_paths):
        points = np.load(path)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.estimate_normals()
        color = colors[i] if colors else default_colors[i % len(default_colors)]
        pcd.paint_uniform_color(color)
        pcds.append(pcd)

    o3d.visualization.draw_geometries(pcds)
    
if __name__ == "__main__":
    # Example usage
    # npy_paths = ["data/my_pointcloud/peg_pointcloud_filtered_scaled.npy", "data/my_pointcloud/peg_pointcloud_filtered_scaled_transform.npy"]
    npy_paths = ["src_demo.npy", "src_transform_demo.npy"]
    visualize_multiple_npy_point_clouds(npy_paths)