import numpy as np
import torch

# np.set_printoptions(threshold=sys.maxsize)

def onehot(data,n):
    buf = np.zeros(data.shape + (n,))
    nmsk = np.arange(data.size)*n + data.ravel()
    buf.ravel()[nmsk-1] = 1
    return buf


