# Green COD

Please download the data from https://hongshuochen.com/GreenCOD/ and put it in the `data` folder.


## Setup
```bash
conda create --name cod
conda activate cod
conda install -c anaconda pip
pip3 install torch torchvision torchaudio
pip install xgboost opencv-python tqdm matplotlib scikit-learn
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    cudf-cu11 dask-cudf-cu11 cuml-cu11 cugraph-cu11 cuspatial-cu11 cuproj-cu11 cuxfilter-cu11 cucim
pip install cupy-cuda11x
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
```

## Run
```bash
python main.py --exp_name EfficientNetB4_416_26 \
    --early_stopping_rounds 10 --gpu_id 1 \
    --batch_size 100 --seed 2023 \
    --scale 416 --feature_shape_1 26 --all_size_1 \
    --num_boost_round 1000 --eta 0.1
```

```bash
python main_load_1xgb.py --exp_name EfficientNetB4_416_26_52 \
    --early_stopping_rounds 10 --gpu_id 1 \
    --batch_size 100 --seed 2024 \
    --scale 416 --feature_shape_1 26 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_416_26/2023-08-27 20:03:55.780469/model/xgboost.pkl" \
    --feature_shape_2 52 --prob_kernel_size_2 19 --all_size_2 \
    --sample_ratio 0.25 --num_boost_round 1000 --eta 0.1
```

```bash
python main_load_2xgb.py --exp_name EfficientNetB4_416_26_52_104 \
    --early_stopping_rounds 10 --gpu_id 1 \
    --batch_size 100 --seed 2025 \
    --scale 416 --feature_shape_1 26 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_416_26/2023-08-27 20:03:55.780469/model/xgboost.pkl" \
    --feature_shape_2 52 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_416_26_52/2023-08-27 20:10:27.608233/model/xgboost.pkl" \
    --feature_shape_3 104 --prob_kernel_size_3 19 --all_size_3 \
    --sample_ratio 0.05 --num_boost_round 1000 --eta 0.1
```

```bash
python main_load_3xgb.py --exp_name EfficientNetB4_416_26_52_104_104 \
    --early_stopping_rounds 10 --gpu_id 1 \
    --batch_size 100 --seed 2026 \
    --scale 416 --feature_shape_1 26 --all_size_1 \
    --model_path_1 "ckpt/EfficientNetB4_416_26/2023-08-27 20:03:55.780469/model/xgboost.pkl" \
    --feature_shape_2 52 --prob_kernel_size_2 19 --all_size_2 \
    --model_path_2 "ckpt/EfficientNetB4_416_26_52/2023-08-27 20:10:27.608233/model/xgboost.pkl" \
    --feature_shape_3 104 --prob_kernel_size_3 19 --all_size_3 \
    --model_path_3 "ckpt/EfficientNetB4_416_26_52_104/2023-08-27 20:54:00.375226/model/xgboost.pkl" \
    --feature_shape_4 104 --prob_kernel_size_4 19 --all_size_4 \
    --sample_ratio 0.05 --num_boost_round 1000 --eta 0.1
```
