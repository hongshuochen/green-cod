import os
import time
import torch
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

class SingleXGBoost:
    def __init__(self, params, num_boost_round, early_stopping_rounds):
        self.params = params
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.children = []
        
    def fit(self, DMatrix_train, DMatrix_val, *args, **kwargs):
        self.evals_result = {}
        watchlist = [(DMatrix_train, 'train'), (DMatrix_val, 'val')]
        self.bst = xgb.train(self.params, DMatrix_train, self.num_boost_round, 
                            evals=watchlist, early_stopping_rounds=self.early_stopping_rounds,
                            evals_result=self.evals_result, verbose_eval=True, *args, **kwargs)
        return self
    
    def predict(self, X, iteration_range=None):
        start = time.time()
        if isinstance(X, np.ndarray) or isinstance(X, torch.Tensor):
            X = xgb.DMatrix(X)
        if iteration_range is None:
            iteration_range = (0, self.bst.best_iteration+1)
        # print(iteration_range)
        output = self.bst.predict(X, iteration_range=iteration_range)
        # print("XGBoost Predict Finished in", time.time() - start, "seconds.")
        return output
    
    def inplace_predict(self, X, iteration_range=None):
        start = time.time()
        if iteration_range is None:
            iteration_range = (0, self.bst.best_iteration+1)
        print(iteration_range)
        output = self.bst.inplace_predict(X, iteration_range=iteration_range)
        print("XGBoost Inplace Predict Finished in", time.time() - start, "seconds.")
        return output

    def cuda(self, gpu_id=0):
        self.bst.set_param({"predictor": "gpu_predictor"})
        self.bst.set_param({"gpu_id": gpu_id})
        
    def cpu(self):
        self.bst.set_param({"predictor": "cpu_predictor"})
    
    def to(self, device):
        if device == 'cuda':
            self.cuda()
        else:
            self.cpu()
        
    def plot_learning_curve(self, eval_metric='logloss', path=None):
        plt.plot(self.evals_result['train'][eval_metric], label='train')
        plt.plot(self.evals_result['val'][eval_metric], label='val')
        plt.xlabel('Iteration')
        plt.ylabel(eval_metric)
        plt.legend()
        if path is not None:
            plt.savefig(path)
            # plt.show()
            plt.close()
        else:
            plt.show()
            
    def get_feature_importance(self, importance_type='gain'):
        feature_importance = self.bst.get_score(importance_type=importance_type)
        feature_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        feature_importance = np.array(feature_importance)
        feature_importance = [int(i[1:]) for i in feature_importance[:, 0]]
        return feature_importance
    
    def get_feature_importance_score(self, importance_type='gain'):
        feature_importance = self.bst.get_score(importance_type=importance_type)
        feature_importance = feature_importance.items()
        feature_importance = [y for x, y  in feature_importance]
        return feature_importance
    
    def get_num_boost_round(self):
        return self.bst.num_boost_round
    
    def save_model(self, model_path):
        start = time.time()
        self.bst.save_model(model_path)
        print("Save XGBoost Finished in", time.time() - start, "seconds.")
        
def plot_distribution(y, prob):
    plt.hist(prob[y==0], bins=100, alpha=0.5, label='0')
    plt.hist(prob[y==1], bins=100, alpha=0.5, label='1')
    plt.legend()
    plt.show()
    
if __name__ == "__main__":
    import numpy as np
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, log_loss
    
    X, y = make_classification(n_samples=1000000, n_features=256, n_informative=10, n_classes=2, class_sep=0.1, random_state=2)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'max_depth': 3,
        'eta': 0.1,
        'nthread': 4,
        'seed': 0,
        'max_delta_step': 1,
    }
    num_boost_round = 10000
    early_stopping_rounds = 100
    
    sxgb = SingleXGBoost(params, num_boost_round, early_stopping_rounds)
    sxgb.fit(X_train, y_train, X_test, y_test)
    sxgb.plot_learning_curve()
    y_pred = sxgb.bst.predict(xgb.DMatrix(X_test))
    plot_distribution(y_test, y_pred)
    # y_pred = np.where(y_pred > 0.5, 1, 0)
    # print(accuracy_score(y_test, y_pred))
    print(log_loss(y_test, y_pred))