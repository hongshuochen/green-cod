import time
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F

from resize import Resize

def image_features(model, images, feature_shape, start_layer, end_layer, all_size=False, equal=False):
    start = time.time()
    features = []
    with torch.no_grad():
        input = images
        for i in range(8):
            input = model.features[i](input)
            if input.shape[-1] != feature_shape:
                output = F.interpolate(input, size=(feature_shape, feature_shape), mode="bilinear")
            else:
                output = input
            if i in range(start_layer, end_layer):
                if all_size or input.shape[-1] == feature_shape or (equal == False and input.shape[-1] >= feature_shape):
                    features.append(output)
    features = torch.cat(features, dim=1)
    features = features.permute(0, 2, 3, 1)
    features = features.reshape(-1, features.shape[-1])
    print("Image features shape:", features.shape)
    print("image_features Finished in", time.time() - start, "seconds.")
    return features.cpu().numpy()
    
# concatenate image features with previous prediction
def prob_features(features, y_prev_pred, prob_kernel_size, prev_feature_shape, feature_shape, prob_only=False):
    start = time.time()
    y_prev_pred = torch.from_numpy(y_prev_pred)
    y_prev_pred = y_prev_pred.reshape(-1, 1, prev_feature_shape, prev_feature_shape)
    y_prev_pred = Resize(size=(feature_shape, feature_shape))(y_prev_pred)
    prev_unfold = nn.Unfold(kernel_size=(prob_kernel_size, prob_kernel_size), padding=prob_kernel_size//2)
    y_prev_pred = prev_unfold(y_prev_pred)
    y_prev_pred = y_prev_pred.reshape(len(y_prev_pred), -1, feature_shape, feature_shape)
    y_prev_pred = y_prev_pred.permute(0, 2, 3, 1)
    y_prev_pred = y_prev_pred.reshape(-1, y_prev_pred.shape[-1])
    y_prev_pred = y_prev_pred.numpy()
        
    # concat features
    if prob_only:
        X_train = y_prev_pred
    else:
        X_train = np.concatenate([features, y_prev_pred], axis=1)

    print("Image + Prediction features shape:", X_train.shape)
    print("prob_features Finished in", time.time() - start, "seconds.")
    return X_train