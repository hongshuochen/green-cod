from argparse import ArgumentParser

# Parse input arguments
def parse_arguments():
    parser = ArgumentParser()
    # General parameters
    parser.add_argument("--gpu_id", default=0, type=int)
    parser.add_argument("--epoch_num", default=1, type=int)
    parser.add_argument("--batch_size", default=40, type=int)
    parser.add_argument("--test_batch_size", default=40, type=int)
    parser.add_argument("--dataset_size", default=4040, type=int)
    parser.add_argument("--train_size", default=3200, type=int)
    parser.add_argument("--scale", default=416, type=int)
    parser.add_argument("--exp_name", default="exp", type=str)
    parser.add_argument("--ckpt_path", default="./ckpt", type=str)
    parser.add_argument("--seed", default=2023, type=int)
    parser.add_argument("--num_workers", default=32, type=int)
    parser.add_argument("--sample_ratio", default=1, type=float)
    parser.add_argument("--infer", action="store_true")
    parser.add_argument("--debug", action="store_true")
    
    # Edge parameters
    parser.add_argument("--type", default="GT", type=str)
    parser.add_argument("--erosion", action="store_true")
    parser.add_argument("--thick", default=1, type=int)
    parser.add_argument("--edge_feature_shape", default=26, type=int)
    
    # Feature Extraction parameters
    parser.add_argument("--feature_shape_1", default=26, type=int)
    parser.add_argument("--prob_only_1", action="store_true") # only probabilities
    parser.add_argument("--equal_1", action="store_true") # only features with the same shape
    parser.add_argument("--all_size_1", action="store_true") # use all features, default only using features larger than and equal to the current size 
    parser.add_argument("--start_layer_1", default=0, type=int)
    parser.add_argument("--end_layer_1", default=8, type=int)
    parser.add_argument("--prob_kernel_size_1", default=3, type=int)
    parser.add_argument("--model_path_1", default="", type=str)
    
    parser.add_argument("--feature_shape_2", default=52, type=int)
    parser.add_argument("--prob_only_2", action="store_true") # only probabilities
    parser.add_argument("--equal_2", action="store_true") # only features with the same shape
    parser.add_argument("--all_size_2", action="store_true") # use all features, default only using features larger than and equal to the current size 
    parser.add_argument("--start_layer_2", default=0, type=int)
    parser.add_argument("--end_layer_2", default=8, type=int)
    parser.add_argument("--prob_kernel_size_2", default=3, type=int)
    parser.add_argument("--model_path_2", default="", type=str)
    
    parser.add_argument("--feature_shape_3", default=52, type=int)
    parser.add_argument("--prob_only_3", action="store_true") # only probabilities
    parser.add_argument("--equal_3", action="store_true") # only features with the same shape
    parser.add_argument("--all_size_3", action="store_true") # use all features, default only using features larger than and equal to the current size 
    parser.add_argument("--start_layer_3", default=0, type=int)
    parser.add_argument("--end_layer_3", default=8, type=int)
    parser.add_argument("--prob_kernel_size_3", default=3, type=int)
    parser.add_argument("--model_path_3", default="", type=str)
    
    parser.add_argument("--feature_shape_4", default=52, type=int)
    parser.add_argument("--prob_only_4", action="store_true") # only probabilities
    parser.add_argument("--equal_4", action="store_true") # only features with the same shape
    parser.add_argument("--all_size_4", action="store_true") # use all features, default only using features larger than and equal to the current size 
    parser.add_argument("--start_layer_4", default=0, type=int)
    parser.add_argument("--end_layer_4", default=8, type=int)
    parser.add_argument("--prob_kernel_size_4", default=3, type=int)
    parser.add_argument("--model_path_4", default="", type=str)

    parser.add_argument("--cur_model_path", default="", type=str)
    
    # XGBoost parameters
    parser.add_argument("--eta", default=0.02, type=float)
    parser.add_argument("--depth", default=8, type=int)
    parser.add_argument("--num_boost_round", default=10000, type=int)
    parser.add_argument("--early_stopping_rounds", default=10, type=int)
    parser.add_argument("--colsample_bytree", default=0.8, type=float)
    parser.add_argument("--max_delta_step", default=0, type=int)
    
    args = parser.parse_args()
    return vars(args)