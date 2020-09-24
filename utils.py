import torch
import matplotlib.pyplot as plt
import numpy as np

def xavier(in_shape):

    val = torch.randn(in_shape) * (1/np.sqrt(in_shape[0]/2))
    return val

def latent_space(mean_layer, std_layer):
    epsilon = torch.randn(std_layer.size())
    return mean_layer + torch.exp(0.5 * std_layer) * epsilon 

if __name__ == "__main__":

    in_shape = [28*28, 512]

    weights = xavier(in_shape)
    print(weights.mean(axis=0))
    #print(weights.std(axis=0))
    plt.hist(np.array(weights), bins=100)
    plt.show()

    print(weights.size())
