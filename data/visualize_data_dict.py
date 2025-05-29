import numpy as np
import open3d as o3d
from typing import List, Optional
from numpy.typing import NDArray
import pickle
from geotransformer.utils.my_data_utils import visualize_multiple_npy_point_clouds, getting_pointcloud_from_transform, calculate_transform_error_average

def main():
    data_dict = pickle.load(open("results/result_simulation_data.pkl", "rb"))
    gt_transform_matrix = data_dict["gt_peg_to_hole_transform_matrix_list"][0]  # 変換行列
    
    original_pcd_peg = data_dict["pointcloud_original_peg"]  # 元のペグの点群
    filtered_pcd_peg = data_dict["pointcloud_filtered_peg"]  # フィルタリング後のペグの点群
    gt_pointcloud_peg = data_dict["gt_pointcloud_filtered_peg_list"][0]  # GTのペグの点群
    transformed_filtered_pcd_peg = getting_pointcloud_from_transform(filtered_pcd_peg, gt_transform_matrix)  # フィルタリング後の点群にGTの変換行列をかけて得られるペグの点群
    original_pcd_hole = data_dict["pointcloud_original_hole_list"][0]  # 元のホールの点群
    filtered_pcd_hole = data_dict["pointcloud_filtered_hole_list"][0]  # フィルタリング後のホールの点群
    # feature_transform_matrix = data_dict["feature_matching_transform_peg_to_peg_matrix_list"][0]  # 特徴マッチングによる変換行列
    # feature_transformed_points = getting_pointcloud_from_transform(data_dict["pointcloud_filtered_peg"], feature_transform_matrix)  # 特徴マッチングから得られた変換行列をかけることによって得られた点群
    # feature_transformed_result = data_dict["feature_matching_transformed_peg_to_peg_pcd_list"][0] # 特徴マッチングの結果から得られた点群
    # icp_transform_matrix = data_dict["icp_transform_peg_to_peg_matrix_list"][0]  # ICPによる変換行列
    # icp_transformed_points = getting_pointcloud_from_transform(feature_transformed_result, icp_transform_matrix)  # ICPから得られた変換行列をかけることによって得られた点群
    # icp_transformed_result = data_dict["icp_transformed_peg_to_peg_pcd_list"][0] # ICPの結果から得られた点群
    # estimated_transform_matrix =  icp_transform_matrix @ feature_transform_matrix # 推定された変換行列
    # transform_error = calculate_translation_and_angle_error_SE2(
    #     gt_transform=gt_transform_matrix,
    #     estimated_transform=estimated_transform_matrix
    # )
    print(f"transform_error=[translation, rotation]: {transform_error}")
    
    # visualize_multiple_npy_point_clouds([gt_pointcloud_peg, feature_transformed_points])
    visualize_multiple_npy_point_clouds([feature_transformed_points, feature_transformed_result])
    visualize_multiple_npy_point_clouds([icp_transformed_points, icp_transformed_result])
    visualize_multiple_npy_point_clouds([feature_transformed_points, getting_pointcloud_from_transform(filtered_pcd_peg, estimated_transform_matrix)])
    # visualize_multiple_npy_point_clouds([gt_pointcloud_peg])    
    visualize_multiple_npy_point_clouds([gt_pointcloud_peg, icp_transformed_result])

if __name__ == "__main__":
    main()
    