import argparse

import torch
import numpy as np
from numpy.typing import NDArray
import pickle

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error
from geotransformer.utils.my_data_utils import visualize_multiple_npy_point_clouds, getting_pointcloud_from_transform, calculate_translation_and_angle_error_SE2

from config import make_cfg
from model import create_model
import open3d as o3d
from typing import List, Optional

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--ref_file", required=True, help="src point cloud numpy file")
    parser.add_argument("--gt_file", required=False, help="ground-truth transformation file")
    parser.add_argument("--weights", required=True, help="model weights file")
    return parser


def load_data(args):
    src_points = np.load(args.src_file) # N x 3
    ref_points = np.load(args.ref_file)
    src_feats = np.ones_like(src_points[:, :1])# N x 1
    ref_feats = np.ones_like(ref_points[:, :1])

    data_dict = {
        "ref_points": ref_points.astype(np.float32),
        "src_points": src_points.astype(np.float32),
        "ref_feats": ref_feats.astype(np.float32),
        "src_feats": src_feats.astype(np.float32),
    }

    if args.gt_file is not None:
        transform = np.load(args.gt_file)
        data_dict["transform"] = transform.astype(np.float32)

    return data_dict

# scale pcd data from 1 meter to 2.5cm
def scale_pcd_data(input_pcd_data:NDArray, scale_factor=40.0) -> NDArray:
    """
    メートル単位の点群データを2.5cm単位にスケーリングして、GeoTransformerで使用できるようにする。
    """
    # ファイル読み込み
    data = input_pcd_data
    
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected Nx3 shape, got {data.shape}")
    
    # スケーリング処理
    scaled_data = data * scale_factor
    
    return scaled_data

# scale pcd data from 2.5cm to 1 meter
def scale_pcd_data_back(input_pcd_data, scale_factor=40.0):
    """
    
    2.5cm単位の点群データをもとの1メートル単位にスケーリングしなおす。
    
    """
    # ファイル読み込み
    data = input_pcd_data
    
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected Nx3 shape, got {data.shape}")
    
    # スケーリング処理
    scaled_data = data / scale_factor
    
    return scaled_data

# scale transform matrix from 2.5cm to 1 meter
def scale_transform_matrix_back(input_transform_matrix:NDArray, scale_factor=40.0) -> NDArray:
    """
    2.5cm単位の変換行列をもとの1メートル単位にスケーリングしなおす。
    
    Args:
        input_transform_matrix (NDArray): 変換行列（4x4のnumpy配列）。
        scale_factor (float): スケーリング係数（デフォルトは40.0）。
        
    Returns:
        NDArray: スケーリングされた変換行列（4x4のnumpy配列）。
    """
    scaled_transform_matrix = input_transform_matrix.copy()
    scaled_transform_matrix[:3, 3] /= scale_factor
    return scaled_transform_matrix



def main():
    cfg = make_cfg()

    # prepare data
    # load from peg and hole pkl data file
    pickle_file_path = "results/result_simulation_data.pkl"
    with open(pickle_file_path, "rb") as f:
        result_data_dict: dict = pickle.load(f)
    
    num_hole_initial_pose = result_data_dict["number_of_initial_hole_pose"]
    feature_transformed_peg_to_hole_pcd_list = [] # list of transformed src points
    feature_transformed_peg_to_peg_pcd_list = [] # list of feature transformed peg to hole point cloud
    feature_matching_transform_peg_to_hole_matrix_list = [] # list of transformed src points
    feature_matching_transform_peg_to_peg_matrix_list = [] # list of feature transform matrix
    
    src_points_origin = result_data_dict["pointcloud_filtered_peg"].astype(np.float32)
    src_points_scaled= scale_pcd_data(src_points_origin, scale_factor=40.0)  # scale to 2.5cm
    src_feats = np.ones_like(src_points_origin[:, :1]).astype(np.float32)
    
    for hole_pose_idx in range(num_hole_initial_pose):
        # load from peg and hole pkl data file
        ref_points_origin = result_data_dict["pointcloud_filtered_hole_list"][hole_pose_idx].astype(np.float32)
        ref_points_scaled= scale_pcd_data(ref_points_origin, scale_factor=40.0)  # scale to 2.5cm
        ref_feats = np.ones_like(ref_points_origin[:, :1]).astype(np.float32)
        gt_transform = result_data_dict["gt_peg_to_hole_transform_matrix_list"][hole_pose_idx].astype(np.float32) # gt transform matrix for peg to hole
        gt_pcd_peg = result_data_dict["gt_pointcloud_filtered_peg_list"][hole_pose_idx].astype(np.float32)  # GT peg point cloud
        gt_pcd_peg_scaled = scale_pcd_data(gt_pcd_peg, scale_factor=40.0)  # scale to 2.5cm
        
            
        # data_dict = load_data(args)
        # peg to hole inference
        data_dict = {
            "ref_points": ref_points_scaled.astype(np.float32),
            "src_points": src_points_scaled.astype(np.float32),
            "ref_feats": ref_feats.astype(np.float32),
            "src_feats": src_feats.astype(np.float32),
            "transform": gt_transform
        }
        neighbor_limits = [38, 36, 36, 38]  # default setting in 3DMatch
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )

        # prepare model
        model = create_model(cfg).cuda()
        weight_path = "weights/geotransformer-3dmatch.pth.tar"
        state_dict = torch.load(weight_path)
        model.load_state_dict(state_dict["model"])
        print("Model loaded successfully.")
        with torch.no_grad():
            model.eval()
            # prediction
            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

            # get results
            src_points_transformed_scaled= output_dict["src_points"]
            src_points_transformed_origin = scale_pcd_data_back(src_points_transformed_scaled, scale_factor=40.0)  # back to 1 meter       
            estimated_transform_scaled = output_dict["estimated_transform"]
            estimated_transform = scale_transform_matrix_back(estimated_transform_scaled, scale_factor=40.0)  # back to 1 meter
            transform = data_dict["transform"]
            feature_transformed_peg_to_hole_pcd_scaled = getting_pointcloud_from_transform(src_points_transformed_scaled, estimated_transform_scaled)  # scale to 2.5cm
            feature_transformed_peg_to_hole_pcd = scale_pcd_data_back(feature_transformed_peg_to_hole_pcd_scaled, scale_factor=40.0)  # back to 1 meter

        feature_transformed_peg_to_hole_pcd_list.append(feature_transformed_peg_to_hole_pcd)
        feature_matching_transform_peg_to_hole_matrix_list.append(estimated_transform)
        print(f"Finished {hole_pose_idx}-th hole inference")
        
        # peg to peg inference
        data_dict = {
            "ref_points": gt_pcd_peg_scaled.astype(np.float32),
            "src_points": src_points_scaled.astype(np.float32),
            "ref_feats": ref_feats.astype(np.float32),
            "src_feats": src_feats.astype(np.float32),
            "transform": gt_transform
        }
        data_dict = registration_collate_fn_stack_mode(
            [data_dict], cfg.backbone.num_stages, cfg.backbone.init_voxel_size, cfg.backbone.init_radius, neighbor_limits
        )
        
        with torch.no_grad():
            model.eval()
            # prediction
            data_dict = to_cuda(data_dict)
            output_dict = model(data_dict)
            data_dict = release_cuda(data_dict)
            output_dict = release_cuda(output_dict)

            # get results
            src_points_transformed_scaled= output_dict["src_points"]
            src_points_transformed_origin = scale_pcd_data_back(src_points_transformed_scaled, scale_factor=40.0)  # back to 1 meter       
            estimated_transform_scaled = output_dict["estimated_transform"]
            estimated_transform = scale_transform_matrix_back(estimated_transform_scaled, scale_factor=40.0)  # back to 1 meter
            transform = data_dict["transform"]
            feature_transformed_peg_to_peg_pcd_scaled = getting_pointcloud_from_transform(src_points_transformed_scaled, estimated_transform_scaled)  # scale to 2.5cm
            feature_transformed_peg_to_peg_pcd = scale_pcd_data_back(feature_transformed_peg_to_peg_pcd_scaled, scale_factor=40.0)  # back to 1 meter
        print(f"Finished {hole_pose_idx}-th peg inference")

        # # visualization
        # ref_pcd = make_open3d_point_cloud(ref_points_origin)
        # ref_pcd.estimate_normals()
        # ref_pcd.paint_uniform_color(get_color("custom_yellow"))
        src_pcd_peg_to_peg = make_open3d_point_cloud(src_points_transformed_origin)
        # src_pcd_peg_to_peg.estimate_normals()
        # src_pcd_peg_to_peg.paint_uniform_color(get_color("custom_blue"))
        # # draw_geometries(ref_pcd, src_pcd_peg_to_peg)
        # # save_point_cloud_to_npy(ref_pcd, "ref_demo.npy")
        src_pcd_peg_to_peg = src_pcd_peg_to_peg.transform(estimated_transform)
        # # save_point_cloud_to_npy(src_pcd_peg_to_peg, "src_transform_demo.npy")
        # # draw_geometries(ref_pcd, src_pcd_peg_to_peg)

        
        feature_transformed_peg_to_peg_pcd_list.append(feature_transformed_peg_to_peg_pcd)
        feature_matching_transform_peg_to_peg_matrix_list.append(estimated_transform)

            
    result_data_dict["feature_matching_transformed_peg_to_peg_pcd_list"] = feature_transformed_peg_to_peg_pcd_list
    result_data_dict["feature_matching_transform_peg_to_peg_matrix_list"] = feature_matching_transform_peg_to_peg_matrix_list
    result_data_dict["feature_matching_transformed_peg_to_hole_pcd_list"] = feature_transformed_peg_to_hole_pcd_list
    result_data_dict["feature_matching_transform_peg_to_hole_matrix_list"] = feature_matching_transform_peg_to_hole_matrix_list
    
    
    with open("results/result_simulation_data_feature.pkl", "wb") as f:
        pickle.dump(result_data_dict, f)

    # compute error
    # rre, rte = compute_registration_error(transform, estimated_transform)
    # print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
