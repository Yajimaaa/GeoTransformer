# GeoTransformerの使い方
## 特徴量抽出のコードの実行のしかた


```bash
CUDA_VISIBLE_DEVICES=0 python experiments/geotransformer.3dmatch.stage4.gse.k3.max.oacl.stage2.sinkhorn/my_demo.py
```
my_demo.pyでは、results/result_simulation_data.pklのデータを自動でロードするようになっていて、このデータのなかから平面除去されたペグとホールのデータの特徴量マッチングを行い、ペグとホールに registration するようになっている。

特徴量マッチングによる変換後の点群と、その変換行列のデータを追加してresults/result_simulation_data_feature.pklに新たに保存する（いまは、検証のためにpeg-to-pegとpeg-to-holeの2つをやっているが将来的にpeg-to-holeのみにしていいかも）