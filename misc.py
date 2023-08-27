import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pickle
import subprocess
import resource
import psutil

class AvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def check_mkdir(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)

def _sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
# save model
def save_model(model, path):
    with open(path, 'wb') as file:
        # Dump the dictionary into the file
        pickle.dump(model, file)
        
# load model
def load_model(path):
    with open(path, 'rb') as file:
        # Load the dictionary back from the pickle file.
        model = pickle.load(file)
    return model

def write(path, content):
    with open(path, 'a') as f:
        f.write(content)

def get_gpu_memory_usage():
    print("GPU Memory Usage:")
    print("GPU allocated memory by torch:", torch.cuda.memory_allocated()/1024**3, "GB")
    print("GPU cached memory by torch:", torch.cuda.memory_reserved()/1024**3, "GB")
    try:
        result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,nounits,noheader'])
        gpu_memory = [int(x) for x in result.decode('utf-8').strip().split('\n')]
        for gpu_id, memory_used in enumerate(gpu_memory):
            print(f"GPU {gpu_id}: Memory Used = {memory_used} MiB")
    except subprocess.CalledProcessError as e:
        print(f"Error while getting GPU memory usage: {e}")
        return None
    
def get_memory_usage():
    # Get memory usage statistics
    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    # Total physical memory (RAM) in bytes
    total_memory = memory.total

    # Currently used memory in bytes
    used_memory = mem_info.rss

    # Currently available memory in bytes
    available_memory = memory.available

    # Memory usage percentage
    memory_usage_percent = memory.percent

    # Memory usage details
    memory_usage = {
        "total": total_memory,
        "used": used_memory,
        "available": available_memory,
        "percent": memory_usage_percent
    }

    print("CPU Memory Usage:")
    print(f"Total memory: {b_to_gb(memory_usage['total'])} GB")
    print(f"Used memory: {b_to_gb(memory_usage['used'])} GB")
    # print(f"Available memory: {b_to_gb(memory_usage['available'])} GB")
    # print(f"Memory usage percentage: {memory_usage['percent']}%")

def peak_memory():
    print('-'*50)
    get_gpu_memory_usage()
    get_memory_usage()
    print("Peak memory:", kb_to_gb(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss), "GB")
    print('-'*50)
    
def kb_to_gb(kb):
    return kb / (1024 * 1024)

def b_to_gb(b):
    return b / (1024 * 1024 * 1024)

def get_train_val_index(n, k):
    index = torch.randperm(n)
    train_index = index[:k]
    val_index = index[k:]
    return train_index, val_index
    
def balanced_sampling(X, y, num_samples_per_class=float('inf')):
    # Get indices of positive and negative samples
    pos_indices = torch.where(y > 0.5)[0]
    neg_indices = torch.where(y <= 0.5)[0]

    # Get the number of positive and negative samples
    num_pos_samples = len(pos_indices)
    num_neg_samples = len(neg_indices)
    num_samples = min(num_pos_samples, num_neg_samples, num_samples_per_class)

    # Sample equal number of positive and negative samples
    pos_samples_indices = torch.randperm(num_pos_samples)[:num_samples]
    pos_samples = X[pos_indices][pos_samples_indices]
    pos_labels = y[pos_indices][pos_samples_indices]
    
    neg_samples_indices = torch.randperm(num_neg_samples)[:num_samples]
    neg_samples = X[neg_indices][neg_samples_indices]
    neg_labels = y[neg_indices][neg_samples_indices]

    # Concatenate and shuffle the samples and labels
    samples = torch.cat((pos_samples, neg_samples), dim=0)
    labels = torch.cat((pos_labels, neg_labels), dim=0)
    indices = np.random.permutation(num_samples*2)
    samples = samples[indices]
    labels = labels[indices]

    return samples, labels

def balanced_sampling_numpy(X, y, num_samples_per_class=float('inf')):
    # Get indices of positive and negative samples
    pos_indices = np.where(y > 0.5)[0]
    neg_indices = np.where(y <= 0.5)[0]
    
    # Get the number of positive and negative samples
    num_pos_samples = len(pos_indices)
    num_neg_samples = len(neg_indices)
    num_samples = min(num_pos_samples, num_neg_samples, num_samples_per_class)
    
    # Sample equal number of positive and negative samples
    pos_samples_indices = torch.randperm(num_pos_samples)[:num_samples]
    neg_samples_indices = torch.randperm(num_neg_samples)[:num_samples]
    pos_samples = X[pos_indices][pos_samples_indices]
    neg_samples = X[neg_indices][neg_samples_indices]
    pos_labels = y[pos_indices][pos_samples_indices]
    neg_labels = y[neg_indices][neg_samples_indices]
    
    # Concatenate and shuffle the samples and labels
    samples = np.concatenate((pos_samples, neg_samples), axis=0)
    labels = np.concatenate((pos_labels, neg_labels), axis=0)
    
    # Shuffle the samples and labels
    indices = np.random.permutation(num_samples*2)
    samples = samples[indices]
    labels = labels[indices]
    return samples, labels

if __name__ == "__main__":
    X = torch.tensor([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
    print(X.shape)
    y = torch.tensor([0.9,0.8,0.7,0.2,0.1,0.15,0.13,0.1, 0.11, 0.16])
    print(balanced_sampling(X, y))