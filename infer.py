import os
import json
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

from misc import load_model
from features import image_features, prob_features
from fastforest import fast
from argparse import ArgumentParser
from sklearn.metrics import mean_absolute_error

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--cupy", action="store_true")
    parser.add_argument("--model_folder", type=str, required=True)
    parser.add_argument("--input_folder", type=str, required=True)
    parser.add_argument("--target_folder", default="", type=str)
    parser.add_argument("--output_folder", type=str, required=True)
    args_infer = parser.parse_args()
    args_infer = vars(args_infer)

    # Determine the device to use
    if args_infer['cpu'] or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{args_infer['gpu_id']}")

    print(f"Using device: {device}")
    print(torch.__version__)

    with open(os.path.join(args_infer["model_folder"], "args.json"), "r") as f:
        args = json.load(f)

    # Load the EfficientNet_B4 model
    print("Load EfficientNet_B4...")
    weights = EfficientNet_B4_Weights.DEFAULT
    model = efficientnet_b4(weights=weights).eval().to(device)

    models = {}
    model_path = os.path.join(args_infer["model_folder"], "model")

    cur = 0
    for i in range(1, 5):
        if args["model_path_" + str(i)] != "":
            models["xgb_" + str(i)] = load_model(os.path.join(args["model_path_" + str(i)]))
            if not args_infer['cpu'] and torch.cuda.is_available():
                models["xgb_" + str(i)] = fast(models["xgb_" + str(i)], model_path, "xgb_" + str(i))
        else:
            args["model_path_" + str(i)] = os.path.join(args_infer["model_folder"], "model", "xgboost.pkl")
            xgb = load_model(os.path.join(args["model_path_" + str(i)]))
            if not args_infer['cpu'] and torch.cuda.is_available():
                xgb = fast(xgb, model_path, "xgb_" + str(i))
            cur = i
            break

    transform = transforms.Compose([
        transforms.Resize((672, 672)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    os.makedirs(args_infer['output_folder'], exist_ok=True)
    image_list = os.listdir(args_infer['input_folder'])
    mae_list = []

    for image_path in tqdm(image_list):
        image = Image.open(os.path.join(args_infer['input_folder'], image_path))
        W, H = image.size 
        image = transform(image)
        image = image.to(device).unsqueeze(0)

        with torch.no_grad():
            features = image_features(model, image, args["feature_shape_1"], args["start_layer_1"], args["end_layer_1"], all_size=args["all_size_1"], equal=args["equal_1"])

            if "xgb_1" in models:
                y_prev_train = models["xgb_1"].predict(features)
                features = image_features(model, image, args["feature_shape_2"], args["start_layer_2"], args["end_layer_2"], all_size=args["all_size_2"], equal=args["equal_2"], cupy=args_infer['cupy'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_2"], args["feature_shape_1"], args["feature_shape_2"], args["prob_only_2"], cupy=args_infer['cupy'])

            if "xgb_2" in models:
                y_prev_train = models["xgb_2"].predict(features)
                features = image_features(model, image, args["feature_shape_3"], args["start_layer_3"], args["end_layer_3"], all_size=args["all_size_3"], equal=args["equal_3"], cupy=args_infer['cupy'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_3"], args["feature_shape_2"], args["feature_shape_3"], args["prob_only_3"], cupy=args_infer['cupy'])

            if "xgb_3" in models:
                y_prev_train = models["xgb_3"].predict(features)
                features = image_features(model, image, args["feature_shape_4"], args["start_layer_4"], args["end_layer_4"], all_size=args["all_size_4"], equal=args["equal_4"], cupy=args_infer['cupy'])
                features = prob_features(features, y_prev_train, args["prob_kernel_size_4"], args["feature_shape_3"], args["feature_shape_4"], args["prob_only_4"], cupy=args_infer['cupy'])

            y_pred = xgb.predict(features)
            y_pred = torch.as_tensor(y_pred).to(device)
            y_pred = y_pred.reshape(-1, 1, args["feature_shape_" + str(cur)], args["feature_shape_" + str(cur)])
            if args_infer["target_folder"] != "":
                target_image = Image.open(os.path.join(args_infer["target_folder"], image_path.replace('.jpg','.png')))
                target_transform = transforms.ToTensor()
                target_image = target_transform(target_image)
            if args_infer["target_folder"] != "":
                y_pred = F.interpolate(y_pred, size=target_image.shape[-2:], mode="bicubic")
            else:
                y_pred = F.interpolate(y_pred, size=(H, W), mode="bicubic")
                
            y_pred = torch.clamp(y_pred, min=0, max=1)
            y_pred = y_pred > 0.5

            y_pred = y_pred.float().squeeze(0)
            to_pil = transforms.ToPILImage()
            y_pred_image = to_pil(y_pred.cpu())  # Move back to CPU for saving as an image
            output_image_path = os.path.join(args_infer['output_folder'], image_path.split('/')[-1].split('.')[0] + '.png')
            y_pred_image.save(output_image_path)
            
            if args_infer["target_folder"] != "":
                mae = mean_absolute_error(target_image.cpu().numpy().flatten(), y_pred.cpu().numpy().flatten())
                mae_list.append(mae)

    if args_infer["target_folder"] != "":
        print(f"Mean Absolute Error: {np.mean(mae_list)}")
    del xgb, models