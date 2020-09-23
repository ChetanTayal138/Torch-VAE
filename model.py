from torch.autograd import Variable
from utils import xavier
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


class Encoder:

    def __init__(self, input_image, weights, biases):
        self.input = Variable(input_image, requires_grad = True)

    def encode(self):
        encoder_layer = self.input * weights["w1"] + biases["b1"]
        encoder_layer = torch.tanh(encoder_layer)
        mean_layer = encoder_layer * weights["w2"] + biases["b2"]
        std_dev_layer = encoder_layer * weights["w3"] + biases["w3"]



if __name__ == "__main__" :

    IMAGE_DIM = 28 
    NN_DIM = 10

    input_image = Variable(torch.randn(10,15))
    w1 = Weights("weight_matrix_encoder_hidden", [IMAGE_DIM, NN_DIM])

    x = w1 * input_image
    print(x.size())

    print(w1._get_weight().mean(axis=1))



