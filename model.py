from torch.autograd import Variable
from utils import xavier, latent_layer
import torch


class Weights:

    def __init__(self, name, dimension, initialization=xavier):
        self.name = name
        self.dimension = dimension
        self.initialization = initialization
        self.weights = None
    
    def __repr__(self):
        return self.name

    def _get_weight(self):
        self.weights = Variable(self.initialization(self.dimension), requires_grad=True)
        return self.weights

    def __mul__(self, other):
        return torch.matmul(self._get_weight(), other)

    def __add__(self, other):
        return torch.add(self._get_weight(), other)


class Encoder:

    def __init__(self, input_image):
        self.input = Variable(input_image, requires_grad = True)
        
    def encode(self, weights, biases):
        encoder_layer = self.input * weights["w1"]._get_weight() + biases["b1"]._get_weight()
        encoder_layer = torch.tanh(encoder_layer)
        mean_layer = encoder_layer * weights["w2"] + biases["b2"]
        std_dev_layer = encoder_layer * weights["w3"] + biases["b3"]

        return mean_layer, std_dev_layer


class Decoder:
    
    def __init__(self, latent_space, weights, biases):
        self.latent_space = latent_space


    def decode(self):
        decoder_layer = self.latent_space * weights["w4"] + biases["b4"]
        decoder_layer = torch.tanh(decoder_layer)
        decoder_output = decoder_layer * weights["w5"] + biases["b5"]
        decoder_output = torch.sigmoid(decoder_layer)

        return decoder_output

if __name__ == "__main__" :

    IMAGE_DIM = 28 
    NN_DIM = 10

    input_image = Variable(torch.randn(10,15))
    w1 = Weights("weight_matrix_encoder_hidden", [IMAGE_DIM, NN_DIM])

    
    print(w1._get_weight().mean(axis=1))




