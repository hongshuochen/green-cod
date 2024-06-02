# Green COD

Please download the data from https://hongshuochen.com/GreenCOD/ and put it in the `data` folder.


## Setup
```bash
conda create --name cod python=3.9
conda activate cod
pip3 install torch torchvision torchaudio
pip install opencv-python tqdm matplotlib scikit-learn
pip install xgboost==1.7.6
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11 dask-cudf-cu11 cuml-cu11 cugraph-cu11 cuspatial-cu11 cuproj-cu11 cuxfilter-cu11 cucim
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

## Inference
```bash 
# If you do not have groundtruth
# 1000T-D3 (1000 trees, depth 3)
python infer.py --input_folder data/TestDataset/COD10K/Imgs \
                --model_folder "ckpt/EfficientNetB4_672_42_42_84_168_1000/2023-10-23 00:30:50.619529" --cupy \
                --output_folder output

# 10000T-D6 (10000 trees, depth 6)
python infer.py --input_folder data/TestDataset/COD10K/Imgs \
                --model_folder "ckpt/EfficientNetB4_672_42_42_84_168_10000_6/2023-10-24 15:34:37.433389" --cupy \
                --output_folder output

# If you do have groundtruth
# 1000T-D3 (1000 trees, depth 3)
python infer.py --input_folder data/TestDataset/COD10K/Imgs \
                --target_folder data/TestDataset/COD10K/GT \
                --model_folder "ckpt/EfficientNetB4_672_42_42_84_168_1000/2023-10-23 00:30:50.619529" --cupy \
                --output_folder output

# 10000T-D6 (10000 trees, depth 6)
python infer.py --input_folder data/TestDataset/COD10K/Imgs \
                --target_folder data/TestDataset/COD10K/GT \
                --model_folder "ckpt/EfficientNetB4_672_42_42_84_168_10000_6/2023-10-24 15:34:37.433389" --cupy \
                --output_folder output
```
```bash
# match the ablation study in paper 
bash run.sh
```

## Train on GPU (Nvidia GeForce RTX 3080)
Please change the model path.
```bash
python main.py --exp_name EfficientNetB4_672_42 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 10 --test_batch_size 10 --seed 2023 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --num_boost_round 1000 --eta 0.3 --sample_ratio 0.1 --depth 3
```

```bash
python main_load_1xgb.py --exp_name EfficientNetB4_672_42_42 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 10 --test_batch_size 10 --seed 2024 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --sample_ratio 0.05 --num_boost_round 1000 --eta 0.3 --depth 3
```

```bash
python main_load_2xgb.py --exp_name EfficientNetB4_672_42_42_84 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 10 --test_batch_size 10 --seed 2025 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_672_42_42/2024-06-01 17:13:52.875350/model/xgboost.pkl" \
    --feature_shape_3 84 --prob_kernel_size_3 19 --all_size_3 \
    --sample_ratio 0.0125 --num_boost_round 1000 --eta 0.3 --depth 3
```

```bash
python main_load_3xgb.py --exp_name EfficientNetB4_672_42_42_84_168 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 10 --test_batch_size 10 --seed 2026 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_672_42_42/2024-06-01 17:13:52.875350/model/xgboost.pkl" \
    --feature_shape_3 84 --prob_kernel_size_3 19 --all_size_3 \
    --model_path_3 "ckpt/EfficientNetB4_672_42_42_84/2024-06-01 17:30:25.872155/model/xgboost.pkl" \
    --feature_shape_4 168 --prob_kernel_size_4 19 --all_size_4 \
    --sample_ratio 0.003125 --num_boost_round 1000 --eta 0.3 --depth 3
```

## Train on GPU (Nvidia A6000, A40)
```bash
python main.py --exp_name EfficientNetB4_672_42 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 100 --seed 2023 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --num_boost_round 1000 --eta 0.3 --depth 3
```

```bash
python main_load_1xgb.py --exp_name EfficientNetB4_672_42_42 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 100 --seed 2024 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --num_boost_round 1000 --eta 0.3 --depth 3
```

```bash
python main_load_2xgb.py --exp_name EfficientNetB4_672_42_42_84 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 100 --seed 2025 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_672_42_42/2024-06-01 17:13:52.875350/model/xgboost.pkl" \
    --feature_shape_3 84 --prob_kernel_size_3 19 --all_size_3 \
    --sample_ratio 0.25 --num_boost_round 1000 --eta 0.3 --depth 3
```

```bash
python main_load_3xgb.py --exp_name EfficientNetB4_672_42_42_84_168 \
    --early_stopping_rounds 100 --gpu_id 0 \
    --batch_size 10 --seed 2026 \
    --scale 672 --feature_shape_1 42 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_672_42/2024-06-01 17:05:30.867269/model/xgboost.pkl" \
    --feature_shape_2 42 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_672_42_42/2024-06-01 17:13:52.875350/model/xgboost.pkl" \
    --feature_shape_3 84 --prob_kernel_size_3 19 --all_size_3 \
    --model_path_3 "ckpt/EfficientNetB4_672_42_42_84/2024-06-01 17:30:25.872155/model/xgboost.pkl" \
    --feature_shape_4 168 --prob_kernel_size_4 19 --all_size_4 \
    --sample_ratio 0.05 --num_boost_round 1000 --eta 0.3 --depth 3
```