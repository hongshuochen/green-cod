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
from pprint import pprint

import torch
from torch import nn
import torch.nn.functional as F
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from sklearn.metrics import (
    log_loss,
    accuracy_score,
    mean_squared_error,
    mean_absolute_error,
)

import joint_transforms
from config import cod_training_root, cod10k_path
from datasets import ImageFolder
from misc import AvgMeter, check_mkdir, peak_memory, get_train_val_index
from misc import save_model, load_model, write
from resize import Resize
from single_xgboost import SingleXGBoost
from arguments import parse_arguments
from features import image_features, prob_features
from fastforest import fast

def feature_extraction(models, args, data_loader, epoch_num=1):
    X_train = None
    y_train = None
    num_samples = epoch_num*len(data_loader.dataset)*int(args["feature_shape_4"]*args["feature_shape_4"]*args['sample_ratio'])
    cur = 0
    for epoch in range(epoch_num):
        for images, targets in tqdm(data_loader):
            print(images.shape, targets.shape)
            images = images.cuda(args["gpu_id"])
            with torch.no_grad():
                features = image_features(models['backbone'], images, args["feature_shape_1"], args["start_layer_1"], args["end_layer_1"], all_size=args['all_size_1'], equal=args['equal_1'])
                y_prev_train = models['prev_sxgb'].predict(features)
                features = image_features(models['backbone'], images, args["feature_shape_2"], args["start_layer_2"], args["end_layer_2"], all_size=args['all_size_2'], equal=args['equal_2'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_2"], args["feature_shape_1"], args["feature_shape_2"], args['prob_only_2'])
                y_prev_train = models['prev_sxgb_2'].predict(features)
                features = image_features(models['backbone'], images, args["feature_shape_3"], args["start_layer_3"], args["end_layer_3"], all_size=args['all_size_3'], equal=args['equal_3'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_3"], args["feature_shape_2"], args["feature_shape_3"], args['prob_only_3'])
                y_prev_train = models['prev_sxgb_3'].predict(features)
                features = image_features(models['backbone'], images, args["feature_shape_4"], args["start_layer_4"], args["end_layer_4"], all_size=args['all_size_4'], equal=args['equal_4'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_4"], args["feature_shape_3"], args["feature_shape_4"], args['prob_only_4'])
                
            targets = Resize(size=(args["feature_shape_4"], args["feature_shape_4"]))(targets)
            targets = targets.flatten().numpy()
            print(features.shape, targets.shape)
            
            if X_train is None:
                X_train = np.zeros((num_samples, features.shape[1]), dtype=np.float32)
                y_train = np.zeros((num_samples), dtype=np.float32)
            
            if args['sample_ratio'] != 1:
                n = int(args["feature_shape_4"]*args["feature_shape_4"]*args['sample_ratio'])*len(images)
                train_samples_indices = torch.randperm(len(features))[:n]
                features = features[train_samples_indices]
                targets = targets[train_samples_indices]
            assert cur+len(features) <= len(X_train)
            assert cur+len(targets) <= len(y_train)
            assert len(features) == len(targets)
            X_train[cur:cur+len(features)] = features
            y_train[cur:cur+len(targets)] = targets
            cur += len(targets)
            if args["debug"]:
                X_train = features
                y_train = targets
                break
        if not args["debug"]:
            assert cur == len(X_train), cur == len(y_train)
        print(X_train.shape, y_train.shape)
    return X_train, y_train

if __name__ == "__main__":
    args = parse_arguments()
    pprint(args)
    
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
        
    if not args["infer"]:
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
        train_index, val_index = get_train_val_index(
            args["dataset_size"], args["train_size"]
        )

        # Prepare Data Set.
        train_set = ImageFolder(
            cod_training_root, joint_transform, img_transform, target_transform, train_index
        )
        val_set = ImageFolder(
            cod_training_root, joint_transform, img_transform, target_transform, val_index
        )
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

        models = {}
        print("Load EfficientNet_B4...")
        weights = EfficientNet_B4_Weights.DEFAULT
        model = efficientnet_b4(weights=weights).eval().cuda(args["gpu_id"])
        models['backbone'] = model
        
        print("Load previous XGboost...")
        models["prev_sxgb"] = load_model(os.path.join(args["model_path_1"]))
        models["prev_sxgb"] = fast(models["prev_sxgb"], model_path, "prev_sxgb")
        models["prev_sxgb_2"] = load_model(os.path.join(args["model_path_2"]))
        models["prev_sxgb_2"] = fast(models["prev_sxgb_2"], model_path, "prev_sxgb_2")
        models["prev_sxgb_3"] = load_model(os.path.join(args["model_path_3"]))
        models["prev_sxgb_3"] = fast(models["prev_sxgb_3"], model_path, "prev_sxgb_3")
        
        print("Get training data...")
        X_train, y_train = feature_extraction(models, args, train_loader, epoch_num=args["epoch_num"])
        print("Create DMatrix_train...")
        DMatrix_train = xgb.DMatrix(X_train, label=y_train)
        del X_train
        
        print("Get validation data...")
        X_val, y_val = feature_extraction(models, args, val_loader, epoch_num=1)
        print("Create DMatrix_val...")
        DMatrix_val = xgb.DMatrix(X_val, label=y_val)
        del X_val
        
        print("Delete models...")
        del model, weights, models
        peak_memory()
        print("Release torch cached memory...")
        torch.cuda.empty_cache()
        peak_memory()

        params = {
            "gpu_id": args["gpu_id"],
            "tree_method": "gpu_hist",
            "objective": "binary:logistic",
            "eval_metric": ["error", "rmse", "logloss", "mae"],
            "max_depth": args["depth"],
            "eta": args["eta"],
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "seed": 0,
            "max_delta_step": args["max_delta_step"],
            # "verbosity": 3,
        }
        pprint(params)
        num_boost_round = args["num_boost_round"]
        if args["debug"]:
            num_boost_round = 100
        early_stopping_rounds = args["early_stopping_rounds"]
        # decay_rate = 0.995
        # scheduler = xgb.callback.LearningRateScheduler(lambda epoch: params["eta"] * decay_rate ** epoch)

        sxgb = SingleXGBoost(params, num_boost_round, early_stopping_rounds)
        peak_memory()
        start = time.time()
        sxgb.fit(DMatrix_train, DMatrix_val)
        print("Finished in", time.time() - start, "seconds.")
        peak_memory()

        # dump
        save_model(sxgb, os.path.join(model_path, "xgboost.pkl"))
        peak_memory()

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
        print(
            "mse_train: ",
            mse_train,
            "mae_train: ",
            mae_train,
            "accuracy_train: ",
            accuracy_train,
            "log_loss: ",
            logloss_train,
        )
        write(
            log_path,
            "mse_train: %f, mae_train: %f, accuracy_train: %f, log_loss: %f\n"
            % (mse_train, mae_train, accuracy_train, logloss_train),
        )
        del DMatrix_train
        peak_memory()
        
        # Validation score
        y_pred_val = sxgb.predict(DMatrix_val)
        mse_val = mean_squared_error(y_val, y_pred_val)
        mae_val = mean_absolute_error(y_val, y_pred_val)
        accuracy_val = accuracy_score(y_val > 0.5, y_pred_val > 0.5)
        logloss_val = logloss(y_val, y_pred_val)
        print(
            "mse_val: ",
            mse_val,
            "mae_val: ",
            mae_val,
            "accuracy_val: ",
            accuracy_val,
            "log_loss: ",
            logloss_val,
        )
        write(
            log_path,
            "mse_val: %f, mae_val: %f, accuracy_val: %f, log_loss: %f\n"
            % (mse_val, mae_val, accuracy_val, logloss_val),
        )
        del DMatrix_val
        peak_memory()

    # Test
    peak_memory()
    print("Load EfficientNet_B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights).eval().cuda(args["gpu_id"])
    models = {}
    models["prev_sxgb"] = load_model(os.path.join(args["model_path_1"]))
    models["prev_sxgb_2"] = load_model(os.path.join(args["model_path_2"]))
    models["prev_sxgb_3"] = load_model(os.path.join(args["model_path_3"]))
    if not args["infer"]:
        sxgb = load_model(os.path.join(model_path, "xgboost.pkl"))
    else:
        sxgb = load_model(os.path.join(args["cur_model_path"]))
    models["prev_sxgb"] = fast(models["prev_sxgb"], model_path, "prev_sxgb")
    models["prev_sxgb_2"] = fast(models["prev_sxgb_2"], model_path, "prev_sxgb_2")
    models["prev_sxgb_3"] = fast(models["prev_sxgb_3"], model_path, "prev_sxgb_3")
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
        cod10k_path, test_joint_transform, test_img_transform, target_transform
    )
    print("Test set: {}".format(test_set.__len__()))
    test_loader = DataLoader(
        test_set,
        batch_size=args["test_batch_size"],
        num_workers=args["num_workers"],
        shuffle=False,
    )
    mae_test = 0
    mse_test = 0
    error_test = 0
    logloss_test = 0
    mae_test_binary = 0
    baseline = 0
    pos_mae = 0
    neg_mae = 0
    pos_len = 0
    neg_len = 0
    mae_test_advanced = 0
    num_images = 0
    prob = []
    for images, targets in tqdm(test_loader):
        num_images += len(images)
        print(images.shape, targets.shape)
        images = images.cuda(args["gpu_id"])
        with torch.no_grad():
            features = image_features(model, images, args["feature_shape_1"], args["start_layer_1"], args["end_layer_1"], all_size=args["all_size_1"], equal=args["equal_1"])
            
            start = time.time()
            y_prev_train = models["prev_sxgb"].predict(features)
            print("XGBoost Predict Finished in", time.time() - start, "seconds.")
            features = image_features(model, images, args["feature_shape_2"], args["start_layer_2"], args["end_layer_2"], all_size=args["all_size_2"], equal=args["equal_2"])
            features = prob_features(features, y_prev_train, args["prob_kernel_size_2"], args["feature_shape_1"], args["feature_shape_2"], args["prob_only_2"])
            
            start = time.time()
            y_prev_train = models["prev_sxgb_2"].predict(features)
            print("XGBoost Predict Finished in", time.time() - start, "seconds.")
            features = image_features(model, images, args["feature_shape_3"], args["start_layer_3"], args["end_layer_3"], all_size=args["all_size_3"], equal=args["equal_3"])
            features = prob_features(features, y_prev_train, args["prob_kernel_size_3"], args["feature_shape_2"], args["feature_shape_3"], args["prob_only_3"])
            
            start = time.time()
            y_prev_train = models["prev_sxgb_3"].predict(features)
            print("XGBoost Predict Finished in", time.time() - start, "seconds.")
            features = image_features(model, images, args["feature_shape_4"], args["start_layer_4"], args["end_layer_4"], all_size=args["all_size_4"], equal=args["equal_4"])
            features = prob_features(features, y_prev_train, args["prob_kernel_size_4"], args["feature_shape_3"], args["feature_shape_4"], args["prob_only_4"])
        
        start = time.time()
        y_pred = sxgb.predict(features)
        print("XGBoost Predict Finished in", time.time() - start, "seconds.")
        y_pred = torch.from_numpy(y_pred)
        print(y_pred.shape)
        y_pred = y_pred.reshape(-1, 1, args["feature_shape_4"], args["feature_shape_4"])
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
        logloss_test += np.sum(-(targets * np.log(y_pred) + (1 - targets) * np.log(1 - y_pred)))
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
    print(
        "mae_test_binary: ", mae_test_binary, "mae_test_advanced: ", mae_test_advanced
    )
    write(
        log_path,
        "baseline: %f, pos_mae: %f, neg_mae: %f, mae_test_binary %f, mae_test_advanced: %f\n"
        % (baseline, pos_mae, neg_mae, mae_test_binary, mae_test_advanced),
    )
    prob = torch.cat(prob, dim=0)
    prob = prob.numpy()
    np.save(os.path.join(save_path, "prob.npy"), prob)
    del sxgb, model, models
    
    print("python " + full_command)
    file.close()