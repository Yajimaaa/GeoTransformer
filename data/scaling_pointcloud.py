import numpy as np
import os

def scale_and_save_npy(input_path, output_path, scale_factor=40.0):
    """
    メートル単位の .npy ファイルを読み込み、スケーリングして保存する。

    Args:
        input_path (str): 入力の .npy ファイルパス（メートル単位）
        output_path (str): スケーリング後に保存する .npy ファイルパス
        scale_factor (float): スケーリング係数（例: 1/0.025 = 40.0）
    """
    # ファイル読み込み
    data = np.load(input_path)
    
    if data.ndim != 2 or data.shape[1] != 3:
        raise ValueError(f"Expected Nx3 shape, got {data.shape}")
    
    # スケーリング処理
    scaled_data = data * scale_factor

    # 出力ディレクトリ作成（必要なら）
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # 保存
    np.save(output_path, scaled_data)
    print(f"Saved scaled data to: {output_path} (scale factor: {scale_factor})")

if __name__ == "__main__":
    input_np_path = "/home/masaruy/workspace/GeoTransformer/data/my_pointcloud/peg_pointcloud_filtered.npy"
    output_np_path = input_np_path.replace(".npy", "_scaled.npy")
    scale_and_save_npy(
        input_path=input_np_path,
        output_path=output_np_path,
        scale_factor=40.0
    )
    for i in range(0, 5):
        input_np_path = f"/home/masaruy/workspace/GeoTransformer/data/my_pointcloud/hole_pointcloud_filtered_{i}.npy"
        output_np_path = input_np_path.replace(".npy", "_scaled.npy")
        scale_and_save_npy(
            input_path=input_np_path,
            output_path=output_np_path,
            scale_factor=40.0
        )
