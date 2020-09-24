import torch
from torch.autograd import Variable

def reconstruction_loss(original_image, reconstructed_image):

    l = original_image * torch.log(reconstructed_image + 1e-10) + (1-original_image) * torch.log(1-reconstructed_image + 1e-10)
    l = -torch.mean(l, axis=1)
    print(l.requires_grad)
    return l

def kl_divergence_loss(mean_layer, std_layer):
    print(mean_layer.requires_grad)    
    print(std_layer.requires_grad)
    l = torch.exp(std_layer) + torch.square(mean_layer) - 1 - std_layer
    l = 0.5 * torch.mean(l, axis=1)
    print(l.requires_grad)
    return l

def network_loss(reconstruction_loss, kl_divergence_loss, alpha, beta):
    return Variable(reconstruction_loss + kl_divergence_loss)

if __name__ == "__main__":

    torch.manual_seed(5)
    x =  torch.rand(1,28*28)
    y = torch.rand(1,28*28)
    print(x)
    print(y)
    print(reconstruction_loss(x, y))

