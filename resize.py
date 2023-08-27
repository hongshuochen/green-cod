import cv2
import torch
import numpy as np
import multiprocessing as mp

class Resize:
    def __init__(self, size, interpolation=cv2.INTER_LANCZOS4, n_jobs=-1) -> None:
        assert isinstance(size, tuple), 'size must be tuple'
        self.size = size
        self.n_jobs = n_jobs
        self.interpolation = interpolation
        if self.n_jobs == -1:
            self.n_jobs = mp.cpu_count() // 2
    
    def _transform(self, input):
        return cv2.resize(input, self.size, interpolation=self.interpolation)
         
    def transform(self, input):
        assert isinstance(input, torch.Tensor), 'input must be torch.Tensor'
        assert len(input.shape) == 4, 'input must be 4D tensor'
        assert input.shape[1] == 3 or input.shape[1] == 1, 'input must be RGB image or grayscale image'
        assert input.dtype == torch.float32, 'input must be float32'
        n, c, h, w = input.shape
        input = input.permute(0, 2, 3, 1).detach().numpy()
        input = input*255
        input = input.astype(np.uint8)
        with mp.Pool(self.n_jobs) as pool:
            resized = pool.map(self._transform, input)
        resized = np.stack(resized, axis=0)
        resized = resized.astype(np.float32)/255
        resized = resized.reshape(n, self.size[0], self.size[1], c)
        resized = torch.from_numpy(resized)
        resized = resized.permute(0, 3, 1, 2)
        return resized
    
    def transform_single_thread(self, input):
        assert isinstance(input, torch.Tensor), 'input must be torch.Tensor'
        assert len(input.shape) == 4, 'input must be 4D tensor'
        assert input.shape[1] == 3 or input.shape[1] == 1, 'input must be RGB image or grayscale image'
        assert input.dtype == torch.float32, 'input must be float32'
        n, c, h, w = input.shape
        input = input.permute(0, 2, 3, 1).detach().numpy()
        input = input*255
        input = input.astype(np.uint8)
        resized = []
        for i in range(len(input)):
            resized.append(self._transform(input[i]))
        resized = np.stack(resized, axis=0)
        resized = resized.astype(np.float32)/255
        resized = resized.reshape(n, self.size[0], self.size[1], c)
        resized = torch.from_numpy(resized)
        resized = resized.permute(0, 3, 1, 2)
        return resized

    def __call__(self, input):
        # return self.transform(input)
        return self.transform_single_thread(input)
    
if __name__ == '__main__':
    import time
    for i in range(10):
        resize = Resize(size=(42, 42))
        images = torch.randn(10000, 1, 168, 168)
        start = time.time()
        resized1 = resize.transform(images)
        print(time.time()-start, "s")
        print(resized1.shape, resized1.dtype)
        
        start = time.time()
        resized2 = resize.transform_single_thread(images)
        print(time.time()-start, "s")
        print(resized2.shape, resized2.dtype)
        assert torch.equal(resized1, resized2)
        print(torch.equal(resized1, resized2))