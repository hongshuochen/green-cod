import os
import sys
import time
import json
import shutil
import datetime
import numpy as np
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
from scipy import ndimage
from pprint import pprint

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms

import joint_transforms
from config import cod_training_root, cod10k_path

from datasets import ImageFolder
from misc import AvgMeter, check_mkdir, peak_memory, get_train_val_index

from single_xgboost import SingleXGBoost
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)
from resize import Resize
from misc import save_model, load_model, write
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights
    
from argparse import ArgumentParser
from fastforest import fast

if __name__ == "__main__":
    parser = ArgumentParser()
    # General parameters
    parser.add_argument("--epoch_num", default=1, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--dataset_size", default=4040, type=int)
    parser.add_argument("--train_size", default=3200, type=int)
    parser.add_argument("--scale", default=416, type=int)
    parser.add_argument("--exp_name", default="exp", type=str)
    parser.add_argument("--ckpt_path", default="./ckpt", type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--sample_ratio", default=1, type=float)
    parser.add_argument("--type", default="GT", type=str)
    parser.add_argument("--erosion", action="store_true")
    parser.add_argument("--thick", default=1, type=int)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--gpu_id", default=0, type=int, help="set the GPU ID")
    
    # Feature Extraction parameters
    parser.add_argument("--feature_shape_1", default=26, type=int)
    parser.add_argument("--start_layer_1", default=0, type=int)
    parser.add_argument("--end_layer_1", default=8, type=int)

    # XGBoost parameters
    parser.add_argument("--depth", default=8, type=int, help="set the max depth")
    parser.add_argument("--max_delta_step", default=0, type=int, help="set the max delta step")
    parser.add_argument("--colsample_bytree", default=0.8, type=float)
    parser.add_argument("--num_boost_round", default=1000, type=int)
    parser.add_argument("--early_stopping_rounds", default=10, type=int)

    args = parser.parse_args()
    args = vars(args)

    # Path
    ckpt_path = args["ckpt_path"]
    exp_name = args["exp_name"]
    
    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    cur = str(datetime.datetime.now())
    save_path = os.path.join(ckpt_path, exp_name, cur)
    check_mkdir(save_path)
    model_path = os.path.join(ckpt_path, exp_name, cur, "model")
    check_mkdir(model_path)

    log_path = os.path.join(save_path, "log.txt")
    shutil.copy2(sys.argv[0], save_path)
    out_path = os.path.join(save_path, "out.txt")
    file = open(out_path, "w")
    sys.stdout = file
    full_command = " ".join(sys.argv)
    print(cur)
    print("python " + full_command)
    print("Exp name: ", exp_name)
    print("Save Path", save_path)
    pprint(args)
    
    if args["debug"]:
        args["epoch_num"] = 1
        args["batch_size"] = 10

    cudnn.benchmark = True
    torch.manual_seed(args["seed"])
    print(torch.__version__)

    with open(os.path.join(save_path, "args.json"), "w") as f:
        json.dump(args, f, indent= 4)
        
    # Transform Data.
    joint_transform = joint_transforms.Compose(
        [
            joint_transforms.RandomHorizontallyFlip(),
            joint_transforms.Resize((args["scale"], args["scale"])),
        ]
    )
    img_transform = transforms.Compose(
        [
            transforms.ColorJitter(
                brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    target_transform = transforms.ToTensor()

    train_index, val_index = get_train_val_index(args["dataset_size"], args["train_size"])

    # Prepare Data Set.
    train_set = ImageFolder(cod_training_root, joint_transform, 
                            img_transform, target_transform, 
                            train_index, type=args["type"])
    val_set = ImageFolder(cod_training_root, joint_transform, 
                          img_transform, target_transform, 
                          val_index, type=args["type"])
    print("Train set: {}".format(train_set.__len__()))
    print("Val set: {}".format(val_set.__len__()))
    train_loader = DataLoader(
        train_set,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=True,
    )

    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights).eval().cuda(args["gpu_id"])
    
    X_train = []
    y_train = []
    for epoch in range(args["epoch_num"]):
        for images, targets in tqdm(train_loader):
            print(images.shape, targets.shape)
            images = images.cuda(args["gpu_id"])
            with torch.no_grad():
                features = []
                input = images
                for i in range(8):
                    input = model.features[i](input)
                    if input.shape[-1] != args["feature_shape_1"]:
                        output = F.interpolate(
                            input,
                            size=(args["feature_shape_1"], args["feature_shape_1"]),
                            mode="bilinear"
                        )
                    else:
                        output = input
                    if i in range(args["start_layer_1"], args["end_layer_1"]):
                        features.append(output)
            features = torch.cat(features, dim=1)
            targets = Resize(size=(args["feature_shape_1"], args["feature_shape_1"]))(targets)
            print(features.shape, targets.shape)
            if args["erosion"]:
                targets = targets > 0
                for i in range(len(targets)):
                    structuring_element = np.ones((2*args["thick"]+1, 2*args["thick"]+1), bool)
                    eroded_mask = ndimage.binary_erosion(targets[i,0], structure=structuring_element)
                    boundary = np.logical_and(targets[i,0], np.logical_not(eroded_mask))
                    targets[i,0] = boundary
            features = features.permute(0, 2, 3, 1)
            features = features.reshape(-1, features.shape[-1])
            targets = targets.flatten()
            features = features.cpu()
            print(features.shape, targets.shape)
            X_train.append(features)
            y_train.append(targets)
            if args["debug"]:
                break
    X_train = torch.cat(X_train, dim=0)
    y_train = torch.cat(y_train, dim=0)
    X_train = X_train.numpy()
    y_train = y_train.numpy()
    if args["sample_ratio"] != 1:
        train_samples_indices = torch.randperm(len(X_train))[:int(len(X_train)*args["sample_ratio"])]
        X_train = X_train[train_samples_indices]
        y_train = y_train[train_samples_indices]
    print(X_train.shape, y_train.shape)

    # USE DFT TO SELECT FEATURES
    # selected_features = dft(X_train)
    # np.save(os.path.join(model_path, "selected_features.npy"), selected_features)
    # X_train = X_train[:, selected_features]
    
    DMatrix_train = xgb.DMatrix(X_train, label=y_train)
    del X_train
    
    X_val = []
    y_val = []
    for images, targets in tqdm(val_loader):
        print(images.shape, targets.shape)
        images = images.cuda(args["gpu_id"])
        with torch.no_grad():
            features = []
            input = images
            for i in range(8):
                input = model.features[i](input)
                if input.shape[-1] != args["feature_shape_1"]:
                    output = F.interpolate(
                        input,
                        size=(args["feature_shape_1"], args["feature_shape_1"]),
                        mode="bilinear"
                    )
                else:
                    output = input
                if i in range(args["start_layer_1"], args["end_layer_1"]):
                    features.append(output)
        features = torch.cat(features, dim=1)
        targets = Resize(size=(args["feature_shape_1"], args["feature_shape_1"]))(targets)
        if args["erosion"]:
            targets = targets > 0
            structuring_element = np.ones((2*args["thick"]+1, 2*args["thick"]+1), bool)
            for i in range(len(targets)):
                eroded_mask = ndimage.binary_erosion(targets[i,0], structure=structuring_element)
                boundary = np.logical_and(targets[i,0], np.logical_not(eroded_mask))
                targets[i,0] = boundary
        print(features.shape, targets.shape)

        features = features.permute(0, 2, 3, 1)
        features = features.reshape(-1, features.shape[-1])
        features = features.cpu()
        targets = targets.flatten()
        print(features.shape, targets.shape)
        X_val.append(features)
        y_val.append(targets)
        if args["debug"]:
            break
    X_val = torch.cat(X_val, dim=0)
    y_val = torch.cat(y_val, dim=0)
    X_val = X_val.numpy()
    y_val = y_val.numpy()
    if args["sample_ratio"] != 1:
        val_samples_indices = torch.randperm(len(X_val))[:int(len(X_val)*args["sample_ratio"])]
        X_val = X_val[val_samples_indices]
        y_val = y_val[val_samples_indices]
    print(X_val.shape, y_val.shape)
    
    DMatrix_val = xgb.DMatrix(X_val, label=y_val)
    del X_val
    
    del model, images, targets, input, output, features
    peak_memory()
    print("Release torch cached memory...")
    torch.cuda.empty_cache()
    peak_memory()
    
    # hyperparameters fine-tuning for XGBoost
    params = {
        "gpu_id": args["gpu_id"],
        "tree_method": "gpu_hist",
        "objective": "binary:logistic",
        "eval_metric": ["error", "rmse", "logloss", "mae"], # change evaluation metrics as your need
        "max_depth": args["depth"],
        "eta": 0.02,
        "subsample": 0.8,
        "colsample_bytree": args["colsample_bytree"],
        "seed": 0,
        "max_delta_step": args["max_delta_step"],
        "sampling_method": args["sampling_method"],
        # "verbosity": 3
    }
    num_boost_round = args["num_boost_round"]
    if args["debug"]:
        num_boost_round = 100
    if args["cpu"]:
        params["tree_method"] = "hist"
    early_stopping_rounds = args["early_stopping_rounds"]
    
    # decay_rate = 0.995
    # scheduler = xgb.callback.LearningRateScheduler(lambda epoch: params["eta"] * decay_rate ** epoch)
    pprint(params)
    sxgb = SingleXGBoost(params, num_boost_round, early_stopping_rounds)

    start = time.time()
    sxgb.fit(DMatrix_train, DMatrix_val)
    print("Finished in", time.time() - start, "seconds.")

    # dump
    save_model(sxgb, os.path.join(model_path, "xgboost.pkl"))

    def logloss(y_true, y_pred):
        if isinstance(y_true, torch.Tensor):
            y_true = y_true.numpy()
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        return np.mean(-(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred)))

    # Plot learning curve
    sxgb.plot_learning_curve(eval_metric="logloss", path=os.path.join(save_path, "logloss.png"))
    sxgb.plot_learning_curve(eval_metric="error", path=os.path.join(save_path, "error.png"))
    sxgb.plot_learning_curve(eval_metric="rmse", path=os.path.join(save_path, "rmse.png"))
    sxgb.plot_learning_curve(eval_metric="mae", path=os.path.join(save_path, "mae.png"))

    # Training score
    y_pred_train = sxgb.predict(DMatrix_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    accuracy_train = accuracy_score(y_train > 0.5, y_pred_train > 0.5)
    logloss_train = logloss(y_train, y_pred_train)
    print("mse_train: ", mse_train,
        "mae_train: ", mae_train,
        "accuracy_train: ", accuracy_train,
        "log_loss: ",logloss_train)
    write(
        log_path,
        "mse_train: %f, mae_train: %f, accuracy_train: %f, log_loss: %f\n"
        % (mse_train, mae_train, accuracy_train, logloss_train),
    )
    
    # Validation score
    y_pred_val = sxgb.predict(DMatrix_val)
    mse_val = mean_squared_error(y_val, y_pred_val)
    mae_val = mean_absolute_error(y_val, y_pred_val)
    accuracy_val = accuracy_score(y_val > 0.5, y_pred_val > 0.5)
    logloss_val = logloss(y_val, y_pred_val)
    print("mse_val: ", mse_val,
        "mae_val: ", mae_val,
        "accuracy_val: ", accuracy_val,
        "log_loss: ", logloss_val)
    write(
        log_path,
        "mse_val: %f, mae_val: %f, accuracy_val: %f, log_loss: %f\n"
        % (mse_val, mae_val, accuracy_val, logloss_val),
    )

    del DMatrix_train, DMatrix_val

    # Test
    peak_memory()
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights).eval().cuda(args["gpu_id"])
    sxgb = load_model(os.path.join(model_path, "xgboost.pkl"))
    sxgb = fast(sxgb, model_path, "sxgb")
    
    test_joint_transform = joint_transforms.Compose(
        [joint_transforms.Resize((args["scale"], args["scale"]))]
    )
    test_img_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    target_transform = transforms.ToTensor()
    test_set = ImageFolder(
        cod10k_path, test_joint_transform, test_img_transform, target_transform, type=args["type"]
    )
    print("Test set: {}".format(test_set.__len__()))
    test_loader = DataLoader(
        test_set,
        batch_size=args["batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
    )
    mse_test = 0
    error_test = 0
    logloss_test = 0
    mae_test = 0
    mae_test_binary = 0
    mae_test_advanced = 0
    baseline = 0
    pos_mae = 0
    neg_mae = 0
    pos_len = 0
    neg_len = 0
    num_images = 0
    prob = []
    for images, targets in tqdm(test_loader):
        num_images += len(images)
        print(images.shape, targets.shape)
        if args["erosion"]:
            targets = targets > 0
            structuring_element = np.ones((2*args["thick"]+1, 2*args["thick"]+1), bool)
            for i in range(len(targets)):
                eroded_mask = ndimage.binary_erosion(targets[i,0],structure=structuring_element)
                boundary = np.logical_and(targets[i,0], np.logical_not(eroded_mask))
                targets[i,0] = boundary
        with torch.no_grad():
            features = []
            input = images
            input = input.cuda(args["gpu_id"])
            for i in range(8):
                input = model.features[i](input)
                print(input.shape)
                if input.shape[-1] != args["feature_shape_1"]:
                    output = F.interpolate(
                        input,
                        size=(args["feature_shape_1"], args["feature_shape_1"]),
                        mode="bilinear"
                    )
                else:
                    output = input
                if i in range(args["start_layer_1"], args["end_layer_1"]):
                    features.append(output)
        features = torch.cat(features, dim=1)
        print(features.shape, targets.shape)

        features = features.permute(0, 2, 3, 1)
        features = features.reshape(-1, features.shape[-1])
        print(features.shape, targets.shape)
        print(features.max(), features.min())
        features = features.cpu().numpy()
        y_pred = sxgb.predict(features)
        y_pred = torch.from_numpy(y_pred)
        print(y_pred.shape)
        y_pred = y_pred.reshape(-1, 1, args["feature_shape_1"], args["feature_shape_1"])
        print(y_pred.shape)
        prob.append(y_pred)
        y_pred = Resize(size=targets.shape[-2:])(y_pred)
        print(y_pred.shape)
        y_pred = y_pred.flatten().numpy()
        print(y_pred.shape)

        targets = targets.flatten().numpy()
        print(targets.shape)
        print(targets.max(), targets.min())
        print(y_pred.max(), y_pred.min())
        y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
        print(y_pred.max(), y_pred.min())

        mae_test += np.sum(np.abs(y_pred - targets))
        mae_test_binary += np.sum(np.abs((y_pred > 0.5) - targets))
        baseline += np.sum(np.abs(targets))
        error_test += np.sum(np.abs((y_pred > 0.5).astype(int) - (targets > 0.5).astype(int)))
        mse_test += np.sum((y_pred - targets) ** 2)
        pos_mae += np.sum(np.abs(y_pred[targets > 0.5] - targets[targets > 0.5]))
        neg_mae += np.sum(np.abs(y_pred[targets <= 0.5] - targets[targets <= 0.5]))
        pos_len += np.sum(targets > 0.5)
        neg_len += np.sum(targets <= 0.5)
        logloss_test += np.sum(
            -(targets * np.log(y_pred) + (1 - targets) * np.log(1 - y_pred))
        )
        y_pred_advanced = np.array(y_pred)
        y_pred_advanced[y_pred_advanced <= 0.5] = 0
        mae_test_advanced += np.sum(np.abs(y_pred_advanced - targets))
        if args["debug"]:
            break
    if not args["debug"]:
        assert num_images == test_set.__len__()
    mae_test /= num_images * args["scale"] * args["scale"]
    mse_test /= num_images * args["scale"] * args["scale"]
    error_test /= num_images * args["scale"] * args["scale"]
    logloss_test /= num_images * args["scale"] * args["scale"]
    mae_test_binary /= num_images * args["scale"] * args["scale"]
    baseline /= num_images * args["scale"] * args["scale"]
    pos_mae /= pos_len
    neg_mae /= neg_len
    mae_test_advanced /= num_images * args["scale"] * args["scale"]
    print(
        "mse_test: ",
        mse_test,
        "mae_test: ",
        mae_test,
        "accuracy_test: ",
        1 - error_test,
        "logloss_test: ",
        logloss_test,
    )
    write(
        log_path,
        "mse_test: %f, mae_test: %f, accuracy_test: %f, logloss_test: %f\n"
        % (mse_test, mae_test, 1 - error_test, logloss_test),
    )
    print("baseline: ", baseline, "pos_mae: ", pos_mae, "neg_mae: ", neg_mae)
    print("mae_test_binary: ", mae_test_binary, "mae_test_advanced: ", mae_test_advanced)
    write(
        log_path,
        "baseline: %f, pos_mae: %f, neg_mae: %f, mae_test_binary %f, mae_test_advanced: %f\n"
        % (baseline, pos_mae, neg_mae, mae_test_binary, mae_test_advanced),
    )
    prob = torch.cat(prob, dim=0)
    prob = prob.numpy()
    np.save(os.path.join(save_path, "prob.npy"), prob)
    del sxgb, model
    
    print("python " + full_command)
    file.close()