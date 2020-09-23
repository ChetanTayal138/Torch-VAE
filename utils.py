import torch
import matplotlib.pyplot as plt
import numpy as np

def xavier(in_shape):

    val = torch.randn(in_shape) * (1/np.sqrt(in_shape[0]/2))
    return val

if __name__ == "__main__":

    in_shape = [28*28, 512]

    weights = xavier(in_shape)
    print(weights.mean(axis=0))
    #print(weights.std(axis=0))
    plt.hist(np.array(weights), bins=100)
    plt.show()

    print(weights.size())
