import numpy as np
import os

def transform_and_save_npy(input_path, output_path, rotation_deg=30, translation_x=0.15):
    """
    点群をZ軸周りに回転し、X方向に平行移動して保存する。

    Args:
        input_path (str): 入力の .npy ファイルパス
        output_path (str): 変換後の .npy ファイルパス
        rotation_deg (float): Z軸まわりの回転角度（度）
        translation_x (float): X方向の平行移動距離（メートル）
    """
    # 点群読み込み
    points = np.load(input_path)
    
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"Expected Nx3 shape, got {points.shape}")

    # 回転角をラジアンに変換
    theta = np.radians(rotation_deg)

    # Z軸周りの回転行列
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta),  np.cos(theta), 0],
        [0,              0,             1]
    ])

    # 回転
    rotated_points = points @ Rz.T

    # 平行移動（X方向にtranslation_x）
    translated_points = rotated_points + np.array([translation_x, 0, 0])

    # 出力ディレクトリがなければ作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    np.save(output_path, translated_points)
    print(f"Transformed point cloud saved to: {output_path}")

# 使用例
if __name__ == "__main__":
    # 入力ファイルと出力ファイルのパスを指定
    input_file = "data/my_pointcloud/peg_pointcloud_filtered_scaled.npy"
    output_file = input_file.replace(".npy", "_transform.npy")
    
    # 点群を変換して保存
    transform_and_save_npy(input_file, output_file)