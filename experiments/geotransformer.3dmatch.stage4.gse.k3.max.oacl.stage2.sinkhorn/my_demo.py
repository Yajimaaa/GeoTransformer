import argparse

import torch
import numpy as np
import pickle

from geotransformer.utils.data import registration_collate_fn_stack_mode
from geotransformer.utils.torch import to_cuda, release_cuda
from geotransformer.utils.open3d import make_open3d_point_cloud, get_color, draw_geometries
from geotransformer.utils.registration import compute_registration_error

from config import make_cfg
from model import create_model


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

def save_point_cloud_to_npy(pcd, filename):
    """
    Open3Dの点群オブジェクトから座標を取り出して.npyに保存
    """
    points = np.asarray(pcd.points)
    np.save(filename, points)
    print(f"Saved point cloud to {filename}")


def main():
    cfg = make_cfg()

    # prepare data
    # load from peg and hole pkl data file
    pickle_file_path = "results/result_simulation_data.pkl"
    with open(pickle_file_path, "rb") as f:
        result_data_dict: dict = pickle.load(f)
    
    num_hole_initial_pose = 200
    src_points_feature_transformed_list = [] # list of transformed src points
    peg_to_hole_feature_matching_transform_matrix_list = [] # list of feature transform matrix
    
    src_points = result_data_dict["pointcloud_filtered_peg"].astype(np.float32)
    src_feats = np.ones_like(src_points[:, :1]).astype(np.float32)
    
    for hole_pose_idx in range(num_hole_initial_pose):
        # load from peg and hole pkl data file
        ref_points = result_data_dict["pointcloud_filtered_hole_list"][hole_pose_idx].astype(np.float32)
        ref_feats = np.ones_like(ref_points[:, :1]).astype(np.float32)
        gt_transform = result_data_dict["gt_peg_to_hole_transform_matrix_list"][hole_pose_idx].astype(np.float32) # gt transform matrix for peg to hole
        
            
        # data_dict = load_data(args)
        data_dict = {
            "ref_points": ref_points.astype(np.float32),
            "src_points": src_points.astype(np.float32),
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
            ref_points = output_dict["ref_points"]
            src_points = output_dict["src_points"]
            estimated_transform = output_dict["estimated_transform"]
            transform = data_dict["transform"]
        print("Finished inference.")

        # visualization
        ref_pcd = make_open3d_point_cloud(ref_points)
        ref_pcd.estimate_normals()
        ref_pcd.paint_uniform_color(get_color("custom_yellow"))
        src_pcd = make_open3d_point_cloud(src_points)
        src_pcd.estimate_normals()
        src_pcd.paint_uniform_color(get_color("custom_blue"))
        # draw_geometries(ref_pcd, src_pcd)
        # save_point_cloud_to_npy(ref_pcd, "ref_demo.npy")
        src_pcd = src_pcd.transform(estimated_transform)
        # save_point_cloud_to_npy(src_pcd, "src_transform_demo.npy")
        # draw_geometries(ref_pcd, src_pcd)
        src_points_feature_transformed = np.asarray(src_pcd.points)
        src_points_feature_transformed_list.append(src_points_feature_transformed)
        peg_to_hole_feature_matching_transform_matrix_list.append(estimated_transform)
        if hole_pose_idx == 0:
            break
            
    result_data_dict["pointcloud_feature_matching_transformed_peg_list"] = src_points_feature_transformed_list
    result_data_dict["peg_to_hole_feature_matching_transform_matrix_list"] = peg_to_hole_feature_matching_transform_matrix_list
    
    with open("results/result_simulation_data.pkl", "wb") as f:
        pickle.dump(result_data_dict, f)

    # compute error
    # rre, rte = compute_registration_error(transform, estimated_transform)
    # print(f"RRE(deg): {rre:.3f}, RTE(m): {rte:.3f}")


if __name__ == "__main__":
    main()
