import os
import numpy as np

T = 32

def split_features(features, T):
    segment_size = len(features) // T
    segments = []
    
    for i in range(T):
        segment = features[i*segment_size:(i+1)*segment_size]
        segments.append(segment.mean(axis=0))
        
    return np.array(segments)


def load_dataset(path):
    data = []
    
    for label_name in ["normal", "anomaly"]:
        label = 0 if label_name == "normal" else 1
        folder = os.path.join(path, label_name)
        
        for file in os.listdir(folder):
            if file.endswith(".npy"):
                features = np.load(os.path.join(folder, file))
                segments = split_features(features, T)
                data.append((segments, label))
                
    return data