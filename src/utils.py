import numpy as np
import torch

def get_batches(data, batch_size=8):
    np.random.shuffle(data)
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        normal = [x[0] for x in batch if x[1] == 0]
        abnormal = [x[0] for x in batch if x[1] == 1]
        
        if len(normal) == 0 or len(abnormal) == 0:
            continue
            
        yield torch.tensor(normal, dtype=torch.float32), \
              torch.tensor(abnormal, dtype=torch.float32)
        

def moving_average(scores, window_size=3):
    return np.convolve(scores, np.ones(window_size)/window_size, mode='same')