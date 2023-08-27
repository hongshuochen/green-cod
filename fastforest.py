import os
from cuml import ForestInference

def fast(model, model_path, name):
    if not os.path.exists(os.path.join(model_path, name + ".model")):
        model.save_model(os.path.join(model_path, name + ".model"))
    model = ForestInference.load(filename=os.path.join(model_path, name + ".model"),
                                            algo='BATCH_TREE_REORG',
                                            model_type='xgboost')
    return model